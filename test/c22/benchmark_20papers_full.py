#!/usr/bin/env python3
"""
Comprehensive benchmark of C22 Profile vs Matrix Profile on 20Papers datasets.

Collects ALL data needed to reproduce PPT-style plots (slides 8-12) on real
datasets instead of synthesized data. Outputs JSON only — no plots generated.
Use plot_20papers_results.py on your local machine to create charts.

Benchmarks performed:
  1. Per-dataset C22 Profile timing (CPU & GPU)
  2. Per-dataset Matrix Profile (SCAMP) timing (CPU & GPU)
  3. GPU vs CPU speedup for both C22 and MP
  4. CPU thread scaling on a representative dataset
  5. Profile comparison: C22 vs MP correlation on each dataset
  6. Anomaly detection: precision of top-k discords against ground-truth labels

Usage (on GPU machine):
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py [options]

  # Full run (CPU + GPU, all benchmarks)
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py --gpu

  # CPU-only quick run
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py --cpu-only

  # Custom settings
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py \\
      --gpu --window-size 50 --threads 16 --repeat 5

Output:
  test/benchmark_20papers_full.json
"""

import sys
import os
import json
import time
import argparse
import multiprocessing
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Build first, then run with PYTHONPATH=build/src/python")
    sys.exit(1)

HAS_GPU = pyscamp.gpu_supported()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WINDOW = 100
REPEAT = 3
MIN_SERIES_LENGTH = 200
THREAD_SCALING_COUNTS = [1, 2, 4, 8, 16]
ANOMALY_TOP_K = 5


def _scamp_mp_is_broken():
    """Detect if pyscamp.selfjoin returns constant profiles (ARM kernel bug)."""
    ts = list(np.sin(np.arange(500, dtype=float) * 0.1) + 0.01 * np.random.randn(500))
    p, _ = pyscamp.selfjoin(ts, 50, threads=1, gpus=[])
    return len(set(float(v) for v in p)) <= 1


SCAMP_MP_BROKEN = _scamp_mp_is_broken()
if SCAMP_MP_BROKEN:
    print("WARNING: pyscamp.selfjoin returns constant profiles on this platform.",
          file=sys.stderr)
    print("         Using numpy brute-force MP for analysis (correlation, anomaly).",
          file=sys.stderr)
    print("         MP timing numbers will still use pyscamp.\n",
          file=sys.stderr)


def compute_mp_brute_force(ts_arr, window):
    """Z-normalized Euclidean distance matrix profile via numpy (float64)."""
    ts = np.asarray(ts_arr, dtype=np.float64)
    n = len(ts) - window + 1
    ez = window // 4
    subs = np.empty((n, window), dtype=np.float64)
    for i in range(n):
        s = ts[i:i + window].copy()
        m, sd = s.mean(), s.std(ddof=0)
        if sd < 1e-12:
            sd = 1.0
        subs[i] = (s - m) / sd
    profile = np.full(n, np.inf)
    index = np.zeros(n, dtype=int)
    chunk = min(n, 2000)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        corr = subs[i0:i1] @ subs.T / window
        dist = np.sqrt(np.maximum(0, 2.0 * window * (1.0 - corr)))
        for li in range(i1 - i0):
            gi = i0 + li
            dist[li, max(0, gi - ez):min(n, gi + ez + 1)] = np.inf
            j = int(np.argmin(dist[li]))
            if dist[li, j] < profile[gi]:
                profile[gi] = dist[li, j]
                index[gi] = j
    return profile, index


def load_timeseries(csv_path):
    """Load time series and labels from CSV. Returns (ts_list, labels_array, error)."""
    try:
        if HAS_PANDAS:
            df = pd.read_csv(csv_path)
            if 'ts' in df.columns:
                ts = df['ts'].dropna().values.astype(np.float64)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return None, None, "No numeric columns found"
                ts = df[numeric_cols[0]].dropna().values.astype(np.float64)

            labels = None
            if 'label' in df.columns:
                labels = df['label'].values.astype(np.float64)
                labels = labels[:len(ts)]
        else:
            import csv as csv_mod
            ts_vals, label_vals = [], []
            with open(csv_path, 'r') as f:
                reader = csv_mod.DictReader(f)
                col_name = 'ts' if 'ts' in reader.fieldnames else reader.fieldnames[0]
                has_label = 'label' in reader.fieldnames
                for row in reader:
                    try:
                        v = float(row[col_name])
                        if not np.isnan(v):
                            ts_vals.append(v)
                            if has_label:
                                label_vals.append(float(row.get('label', 0)))
                    except (ValueError, KeyError):
                        continue
            ts = np.array(ts_vals, dtype=np.float64)
            labels = np.array(label_vals, dtype=np.float64) if label_vals else None

        if len(ts) < MIN_SERIES_LENGTH:
            return None, None, f"Too short ({len(ts)} < {MIN_SERIES_LENGTH})"

        return ts.tolist(), labels, None
    except Exception as e:
        return None, None, str(e)


def bench(func, *args, repeat=REPEAT, **kwargs):
    """Run func repeat times, return min elapsed time in seconds."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"    BENCH ERROR: {e}", file=sys.stderr)
            return None, None
    return min(times) if times else None, result


def throughput(n_subseq, time_s):
    """Million pairwise comparisons per second."""
    if time_s is None or time_s <= 0:
        return None
    return (n_subseq * n_subseq / time_s) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_one_dataset(ts, labels, dataset_name, window, num_threads,
                          use_gpu, repeat):
    """Run all benchmarks on a single dataset. Returns dict of results."""
    n = len(ts)
    n_sub = n - window + 1
    result = {
        'dataset': dataset_name,
        'series_length': n,
        'window_size': window,
        'n_subsequences': n_sub,
    }

    # ── C22 Profile CPU ──
    print(f"    C22 CPU ({num_threads} threads)...", end=" ", flush=True)
    t_c22_cpu, (c22_profile_cpu, c22_index_cpu) = bench(
        pyscamp.selfjoin_c22, ts, window, threads=num_threads, gpu=False,
        repeat=repeat)
    if t_c22_cpu is not None:
        result['c22_cpu'] = {
            'time_s': t_c22_cpu,
            'throughput_mpairs_s': throughput(n_sub, t_c22_cpu),
        }
        print(f"{t_c22_cpu:.3f}s")
    else:
        result['c22_cpu'] = {'error': 'failed'}
        print("FAILED")

    # ── C22 Profile GPU ──
    if use_gpu:
        print(f"    C22 GPU...", end=" ", flush=True)
        t_c22_gpu, (c22_profile_gpu, c22_index_gpu) = bench(
            pyscamp.selfjoin_c22, ts, window, threads=num_threads, gpu=True,
            repeat=repeat)
        if t_c22_gpu is not None:
            result['c22_gpu'] = {
                'time_s': t_c22_gpu,
                'throughput_mpairs_s': throughput(n_sub, t_c22_gpu),
            }
            print(f"{t_c22_gpu:.3f}s")
            if 'time_s' in result.get('c22_cpu', {}):
                result['c22_gpu_speedup'] = result['c22_cpu']['time_s'] / t_c22_gpu
        else:
            result['c22_gpu'] = {'error': 'failed'}
            print("FAILED")

    # ── Matrix Profile (SCAMP) CPU ──
    print(f"    MP  CPU ({num_threads} threads)...", end=" ", flush=True)
    if use_gpu:
        t_mp_cpu, (mp_profile_cpu, mp_index_cpu) = bench(
            pyscamp.selfjoin, ts, window, threads=num_threads, gpus=[],
            repeat=repeat)
    else:
        t_mp_cpu, (mp_profile_cpu, mp_index_cpu) = bench(
            pyscamp.selfjoin, ts, window, threads=num_threads, gpus=[],
            repeat=repeat)
    if t_mp_cpu is not None:
        result['mp_cpu'] = {
            'time_s': t_mp_cpu,
            'throughput_mpairs_s': throughput(n_sub, t_mp_cpu),
        }
        print(f"{t_mp_cpu:.3f}s")
    else:
        result['mp_cpu'] = {'error': 'failed'}
        print("FAILED")

    # ── Matrix Profile (SCAMP) GPU ──
    if use_gpu:
        print(f"    MP  GPU...", end=" ", flush=True)
        t_mp_gpu, (mp_profile_gpu, mp_index_gpu) = bench(
            pyscamp.selfjoin, ts, window, gpus=[0], threads=num_threads,
            repeat=repeat)
        if t_mp_gpu is not None:
            result['mp_gpu'] = {
                'time_s': t_mp_gpu,
                'throughput_mpairs_s': throughput(n_sub, t_mp_gpu),
            }
            print(f"{t_mp_gpu:.3f}s")
            if 'time_s' in result.get('mp_cpu', {}):
                result['mp_gpu_speedup'] = result['mp_cpu']['time_s'] / t_mp_gpu
        else:
            result['mp_gpu'] = {'error': 'failed'}
            print("FAILED")

    # ── Profile comparison (C22 vs MP) ──
    c22_p = np.array(c22_profile_cpu, dtype=np.float64) if c22_profile_cpu is not None else None
    mp_p = np.array(mp_profile_cpu, dtype=np.float64) if mp_profile_cpu is not None else None

    # Fall back to brute-force MP if SCAMP kernel is broken (ARM)
    if SCAMP_MP_BROKEN and c22_p is not None:
        print(f"    Brute-force MP (for analysis)...", end=" ", flush=True)
        mp_p, mp_index_cpu = compute_mp_brute_force(ts, window)
        print(f"done (range {mp_p.min():.2f}–{mp_p.max():.2f})")

    if c22_p is not None and mp_p is not None:
        min_len = min(len(mp_p), len(c22_p))
        _mp = mp_p[:min_len].copy()
        _c22 = c22_p[:min_len].copy()
        valid = np.isfinite(_mp) & np.isfinite(_c22)
        _mp = _mp[valid]
        _c22 = _c22[valid]

        if len(_mp) > 10:
            mp_range = _mp.max() - _mp.min()
            c22_range = _c22.max() - _c22.min()
            if mp_range > 1e-12 and c22_range > 1e-12:
                mp_norm = (_mp - _mp.min()) / mp_range
                c22_inv = -_c22
                c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min())
                pearson_r = float(np.corrcoef(mp_norm, c22_norm)[0, 1])
                if np.isfinite(pearson_r):
                    result['profile_correlation'] = pearson_r
                    print(f"    Profile correlation (Pearson r): {pearson_r:.4f}")
                else:
                    print(f"    Profile correlation: could not compute")
            else:
                print(f"    Profile correlation: constant profile(s), skipped")
        else:
            print(f"    Profile correlation: too few valid points, skipped")

    # ── Anomaly detection evaluation ──
    if labels is not None and c22_p is not None and mp_p is not None:
        label_arr = np.array(labels[:n_sub], dtype=np.float64) if len(labels) >= n_sub else None
        if label_arr is None and len(labels) >= n:
            label_arr = labels[:n_sub]

        if label_arr is not None and np.any(label_arr > 0):
            anomaly_indices = set(np.where(label_arr > 0)[0].tolist())
            n_anomalies = len(anomaly_indices)
            ez = window // 4

            def top_k_discords(profile, k, mode='max'):
                if mode == 'max':
                    order = np.argsort(-profile)
                else:
                    order = np.argsort(profile)
                selected = []
                used = set()
                for idx in order:
                    if len(selected) >= k:
                        break
                    idx = int(idx)
                    if any(abs(idx - u) <= ez for u in used):
                        continue
                    selected.append(idx)
                    used.add(idx)
                return selected

            mp_discords = top_k_discords(mp_p, ANOMALY_TOP_K, 'max')
            c22_discords = top_k_discords(c22_p, ANOMALY_TOP_K, 'min')

            def hits_in_window(candidates, truth, tol):
                """How many candidates fall within `tol` of any truth index."""
                count = 0
                for c in candidates:
                    for off in range(-tol, tol + 1):
                        if (c + off) in truth:
                            count += 1
                            break
                return count

            tol = window
            mp_hits = hits_in_window(mp_discords, anomaly_indices, tol)
            c22_hits = hits_in_window(c22_discords, anomaly_indices, tol)

            result['anomaly_detection'] = {
                'has_labels': True,
                'n_anomaly_points': n_anomalies,
                'top_k': ANOMALY_TOP_K,
                'tolerance': tol,
                'mp_discord_indices': mp_discords,
                'c22_discord_indices': c22_discords,
                'mp_hits': mp_hits,
                'c22_hits': c22_hits,
                'mp_precision_at_k': mp_hits / ANOMALY_TOP_K,
                'c22_precision_at_k': c22_hits / ANOMALY_TOP_K,
            }
            print(f"    Anomaly detection (top-{ANOMALY_TOP_K}): "
                  f"MP {mp_hits}/{ANOMALY_TOP_K}, C22 {c22_hits}/{ANOMALY_TOP_K}")
        else:
            result['anomaly_detection'] = {'has_labels': False}
    else:
        result['anomaly_detection'] = {'has_labels': False}

    # ── C22 vs MP motif info ──
    if c22_p is not None and mp_p is not None:
        c22_idx = np.array(c22_index_cpu) if c22_index_cpu is not None else None
        mp_idx = np.array(mp_index_cpu) if mp_index_cpu is not None else None

        if c22_idx is not None and mp_idx is not None:
            ts_arr = np.array(ts, dtype=np.float64)
            c22_top = int(np.argmax(c22_p))
            c22_match = int(c22_idx[c22_top])
            mp_top = int(np.argmin(mp_p))
            mp_match = int(mp_idx[mp_top])

            def pair_corr(arr, i, j, w):
                a = arr[i:i + w].copy()
                b = arr[j:j + w].copy()
                a = (a - a.mean()) / (a.std() + 1e-10)
                b = (b - b.mean()) / (b.std() + 1e-10)
                return float(np.corrcoef(a, b)[0, 1])

            result['motif_info'] = {
                'c22_top_motif_idx': c22_top,
                'c22_top_motif_match': c22_match,
                'c22_top_motif_dot': float(c22_p[c22_top]),
                'c22_motif_pair_pearson_r': pair_corr(ts_arr, c22_top, c22_match, window),
                'mp_top_motif_idx': mp_top,
                'mp_top_motif_match': mp_match,
                'mp_top_motif_distance': float(mp_p[mp_top]),
                'mp_motif_pair_pearson_r': pair_corr(ts_arr, mp_top, mp_match, window),
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Thread scaling benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_thread_scaling(ts, dataset_name, window, thread_counts, use_gpu, repeat):
    """Measure C22 Profile CPU performance across different thread counts."""
    print(f"\n  Thread scaling on '{dataset_name}' (N={len(ts)}, w={window})")
    n_sub = len(ts) - window + 1

    scaling = {
        'dataset': dataset_name,
        'series_length': len(ts),
        'window_size': window,
        'n_subsequences': n_sub,
        'thread_counts': [],
        'c22_times': [],
        'c22_throughputs': [],
        'c22_speedups': [],
    }
    base_time = None

    for tc in thread_counts:
        if tc > multiprocessing.cpu_count():
            continue
        t, _ = bench(pyscamp.selfjoin_c22, ts, window, threads=tc, gpu=False,
                     repeat=repeat)
        if t is None:
            continue
        if base_time is None:
            base_time = t
        tp = throughput(n_sub, t)
        speedup = base_time / t

        scaling['thread_counts'].append(tc)
        scaling['c22_times'].append(t)
        scaling['c22_throughputs'].append(tp)
        scaling['c22_speedups'].append(speedup)
        print(f"    {tc:>2} threads: {t:.3f}s  {tp:.1f} M pairs/s  {speedup:.2f}x")

    if use_gpu:
        print(f"    GPU...", end=" ", flush=True)
        t_gpu, _ = bench(pyscamp.selfjoin_c22, ts, window,
                         threads=max(scaling['thread_counts'] or [1]),
                         gpu=True, repeat=repeat)
        if t_gpu is not None:
            scaling['gpu_time'] = t_gpu
            scaling['gpu_throughput'] = throughput(n_sub, t_gpu)
            if base_time:
                scaling['gpu_speedup_vs_1thread'] = base_time / t_gpu
            if scaling['c22_times']:
                scaling['gpu_speedup_vs_all_cpu'] = scaling['c22_times'][-1] / t_gpu
            print(f"{t_gpu:.3f}s")

    return scaling


# ─────────────────────────────────────────────────────────────────────────────
# MP vs C22 throughput on same datasets (varying size for scaling curve)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_scaling_by_size(datasets_by_length, window, num_threads, use_gpu, repeat):
    """Throughput vs series length for both C22 and MP, using real data."""
    print(f"\n  Scaling by size (w={window}, {num_threads} threads)")
    scaling = {
        'window_size': window,
        'threads': num_threads,
        'sizes': [],
        'dataset_names': [],
        'c22_cpu_times': [],
        'c22_cpu_throughputs': [],
        'mp_cpu_times': [],
        'mp_cpu_throughputs': [],
    }
    if use_gpu:
        scaling.update({
            'c22_gpu_times': [],
            'c22_gpu_throughputs': [],
            'mp_gpu_times': [],
            'mp_gpu_throughputs': [],
        })

    for name, ts in datasets_by_length:
        n_sub = len(ts) - window + 1

        t_c22, _ = bench(pyscamp.selfjoin_c22, ts, window, threads=num_threads,
                         gpu=False, repeat=repeat)
        t_mp, _ = bench(pyscamp.selfjoin, ts, window, threads=num_threads,
                        gpus=[], repeat=repeat)

        if t_c22 is None or t_mp is None:
            continue

        scaling['sizes'].append(len(ts))
        scaling['dataset_names'].append(name)
        scaling['c22_cpu_times'].append(t_c22)
        scaling['c22_cpu_throughputs'].append(throughput(n_sub, t_c22))
        scaling['mp_cpu_times'].append(t_mp)
        scaling['mp_cpu_throughputs'].append(throughput(n_sub, t_mp))

        print(f"    {name[:40]:<40} N={len(ts):>6}  "
              f"C22={t_c22:.3f}s  MP={t_mp:.3f}s")

        if use_gpu:
            t_c22_g, _ = bench(pyscamp.selfjoin_c22, ts, window,
                               threads=num_threads, gpu=True, repeat=repeat)
            t_mp_g, _ = bench(pyscamp.selfjoin, ts, window, gpus=[0],
                              threads=num_threads, repeat=repeat)
            scaling['c22_gpu_times'].append(t_c22_g if t_c22_g else None)
            scaling['c22_gpu_throughputs'].append(
                throughput(n_sub, t_c22_g) if t_c22_g else None)
            scaling['mp_gpu_times'].append(t_mp_g if t_mp_g else None)
            scaling['mp_gpu_throughputs'].append(
                throughput(n_sub, t_mp_g) if t_mp_g else None)

    return scaling


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive benchmark: C22 vs MP on 20Papers datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CPU-only run
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py

  # With GPU
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py --gpu

  # Custom settings
  PYTHONPATH=build/src/python python3 test/c22/benchmark_20papers_full.py \\
      --gpu --window-size 50 --threads 16 --repeat 5
        """)

    parser.add_argument('--datasets-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', '..', '20Papers '),
                        help='Directory containing CSV datasets (default: ../20Papers /)')
    parser.add_argument('--window-size', '-m', type=int, default=DEFAULT_WINDOW,
                        help=f'Window size (default: {DEFAULT_WINDOW})')
    parser.add_argument('--gpu', action='store_true',
                        help='Also benchmark GPU')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Force CPU only (overrides --gpu)')
    parser.add_argument('--threads', type=int, default=None,
                        help='Number of CPU threads (default: auto)')
    parser.add_argument('--repeat', type=int, default=REPEAT,
                        help=f'Repetitions per measurement, take min (default: {REPEAT})')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'benchmark_20papers_full.json'),
                        help='Output JSON file')
    parser.add_argument('--skip-thread-scaling', action='store_true',
                        help='Skip thread scaling benchmark')
    parser.add_argument('--skip-scaling-by-size', action='store_true',
                        help='Skip scaling-by-size benchmark')
    parser.add_argument('--stdout', action='store_true',
                        help='Print JSON to stdout (for copy-paste from remote host)')

    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu_only and HAS_GPU
    num_threads = args.threads or multiprocessing.cpu_count()

    # When --stdout is used, redirect all progress logging to stderr
    # so stdout contains only clean JSON for copy-paste.
    real_stdout = sys.stdout
    if args.stdout:
        sys.stdout = sys.stderr

    print("=" * 72)
    print("  C22 vs Matrix Profile — 20Papers Dataset Benchmark")
    print("=" * 72)
    print(f"  Datasets dir : {args.datasets_dir}")
    print(f"  Window size  : {args.window_size}")
    print(f"  CPU threads  : {num_threads}")
    print(f"  GPU          : {'YES' if use_gpu else 'NO'}")
    print(f"  Repeat       : {args.repeat}")
    print(f"  Output       : {args.output}")
    print("=" * 72)

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        print(f"ERROR: Directory not found: {datasets_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(datasets_dir.glob('*.csv'))
    if not csv_files:
        print(f"ERROR: No CSV files found in {datasets_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Found {len(csv_files)} CSV files\n")

    # ── Load all datasets ──
    loaded = []
    for csv_path in csv_files:
        name = csv_path.stem
        ts, labels, err = load_timeseries(csv_path)
        if ts is None:
            print(f"  SKIP {name}: {err}")
            continue
        loaded.append((name, ts, labels))
        print(f"  Loaded {name}: N={len(ts)}")

    if not loaded:
        print("ERROR: No valid datasets loaded", file=sys.stderr)
        sys.exit(1)

    # ── Per-dataset benchmarks ──
    all_results = {
        'config': {
            'window_size': args.window_size,
            'cpu_threads': num_threads,
            'gpu_enabled': use_gpu,
            'gpu_supported': HAS_GPU,
            'repeat': args.repeat,
            'n_datasets': len(loaded),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'per_dataset': [],
    }

    for i, (name, ts, labels) in enumerate(loaded):
        print(f"\n{'─' * 72}")
        print(f"  [{i + 1}/{len(loaded)}] {name}  (N={len(ts)})")
        print(f"{'─' * 72}")
        r = benchmark_one_dataset(ts, labels, name, args.window_size,
                                  num_threads, use_gpu, args.repeat)
        all_results['per_dataset'].append(r)

    # ── Thread scaling ──
    if not args.skip_thread_scaling:
        sorted_by_len = sorted(loaded, key=lambda x: len(x[1]))
        mid = len(sorted_by_len) // 2
        rep_name, rep_ts, _ = sorted_by_len[mid]
        tc_list = [t for t in THREAD_SCALING_COUNTS if t <= multiprocessing.cpu_count()]
        all_results['thread_scaling'] = benchmark_thread_scaling(
            rep_ts, rep_name, args.window_size, tc_list, use_gpu, args.repeat)

    # ── Scaling by size ──
    if not args.skip_scaling_by_size:
        sorted_datasets = sorted(loaded, key=lambda x: len(x[1]))
        ds_pairs = [(name, ts) for name, ts, _ in sorted_datasets]
        all_results['scaling_by_size'] = benchmark_scaling_by_size(
            ds_pairs, args.window_size, num_threads, use_gpu, args.repeat)

    # ── Compute summary statistics ──
    per_ds = all_results['per_dataset']

    c22_speedups = [d['c22_gpu_speedup'] for d in per_ds if 'c22_gpu_speedup' in d]
    mp_speedups = [d['mp_gpu_speedup'] for d in per_ds if 'mp_gpu_speedup' in d]
    correlations = [d['profile_correlation'] for d in per_ds if 'profile_correlation' in d]

    anom_ds = [d for d in per_ds
               if d.get('anomaly_detection', {}).get('has_labels')]
    mp_anom_prec = [d['anomaly_detection']['mp_precision_at_k'] for d in anom_ds]
    c22_anom_prec = [d['anomaly_detection']['c22_precision_at_k'] for d in anom_ds]

    summary = {}
    if c22_speedups:
        summary['c22_gpu_speedup_mean'] = float(np.mean(c22_speedups))
        summary['c22_gpu_speedup_median'] = float(np.median(c22_speedups))
        summary['c22_gpu_speedup_min'] = float(np.min(c22_speedups))
        summary['c22_gpu_speedup_max'] = float(np.max(c22_speedups))
    if mp_speedups:
        summary['mp_gpu_speedup_mean'] = float(np.mean(mp_speedups))
        summary['mp_gpu_speedup_median'] = float(np.median(mp_speedups))
        summary['mp_gpu_speedup_min'] = float(np.min(mp_speedups))
        summary['mp_gpu_speedup_max'] = float(np.max(mp_speedups))
    if correlations:
        summary['profile_correlation_mean'] = float(np.mean(correlations))
        summary['profile_correlation_median'] = float(np.median(correlations))
        summary['profile_correlation_min'] = float(np.min(correlations))
        summary['profile_correlation_max'] = float(np.max(correlations))
    if mp_anom_prec:
        summary['mp_anomaly_precision_mean'] = float(np.mean(mp_anom_prec))
        summary['c22_anomaly_precision_mean'] = float(np.mean(c22_anom_prec))
        summary['n_datasets_with_labels'] = len(anom_ds)

    all_results['summary'] = summary

    # ── Write JSON ──
    json_str = json.dumps(all_results, indent=2, default=lambda o: None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(json_str)

    if args.stdout:
        print("\n===JSON_START===")
        print(json_str, file=real_stdout)
        print("===JSON_END===")

    # ── Print summary ──
    print(f"\n{'=' * 72}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Datasets benchmarked: {len(per_ds)}")
    if c22_speedups:
        print(f"  C22 GPU speedup: mean={np.mean(c22_speedups):.2f}x, "
              f"range=[{np.min(c22_speedups):.2f}x, {np.max(c22_speedups):.2f}x]")
    if mp_speedups:
        print(f"  MP  GPU speedup: mean={np.mean(mp_speedups):.2f}x, "
              f"range=[{np.min(mp_speedups):.2f}x, {np.max(mp_speedups):.2f}x]")
    if correlations:
        print(f"  Profile correlation: mean={np.mean(correlations):.3f}, "
              f"range=[{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
    if mp_anom_prec:
        print(f"  Anomaly detection ({len(anom_ds)} datasets with labels):")
        print(f"    MP  precision@{ANOMALY_TOP_K}: {np.mean(mp_anom_prec):.2f}")
        print(f"    C22 precision@{ANOMALY_TOP_K}: {np.mean(c22_anom_prec):.2f}")
    print(f"\n  Results saved to: {output_path}")
    if args.stdout:
        print(f"  JSON printed to stdout — copy everything between "
              f"===JSON_START=== and ===JSON_END===")
    print(f"  Plot with: python3 test/plot_20papers_results.py -i {output_path}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
