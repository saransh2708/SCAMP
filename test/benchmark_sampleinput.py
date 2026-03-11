#!/usr/bin/env python3
"""
Benchmark C22 Profile vs Matrix Profile on SampleInput datasets.

Tests on the SCAMP paper-style data (randomlist, randomwalk, earthquake)
at sizes from 8K to 2M, matching Table 6 of the SCAMP camera-ready paper.
Outputs JSON for local plotting via plot_20papers_results.py.

Usage:
  PYTHONPATH=build/src/python python3 test/benchmark_sampleinput.py --gpu
  PYTHONPATH=build/src/python python3 test/benchmark_sampleinput.py --cpu-only
"""

import sys
import os
import json
import time
import argparse
import multiprocessing
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Build first, then run with PYTHONPATH=build/src/python")
    sys.exit(1)

HAS_GPU = pyscamp.gpu_supported()
SAMPLE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "SampleInput"

DEFAULT_WINDOW = 100
REPEAT = 3
THREAD_SCALING_COUNTS = [1, 2, 4, 8, 16]

DATASETS = [
    ("randomlist8K",     "randomlist8K.txt",     8192),
    ("randomlist16K",    "randomlist16K.txt",     16384),
    ("randomlist32K",    "randomlist32K.txt",     32768),
    ("randomlist64K",    "randomlist64K.txt",     65536),
    ("randomlist128K",   "randomlist128K.txt",    131072),
    ("randomlist256K",   "randomlist256K.txt",    262144),
    ("randomlist512K",   "randomlist512K.txt",    524288),
    ("randomlist1M",     "randomlist1M.txt",      1048576),
    ("randomlist2M",     "randomlist2M.txt",      2097152),
    ("randomwalk8K",     "randomwalk8K.txt",      8192),
    ("randomwalk16K",    "randomwalk16K.txt",     16384),
    ("randomwalk32K",    "randomwalk32K.txt",     32768),
    ("randomwalk64K",    "randomwalk64K.txt",     65536),
    ("randomwalk512K",   "randomwalk512K.txt",    524288),
    ("randomwalk1M",     "randomwalk1M.txt",      1048576),
    ("earthquake95K",    "earthquake_precision_test100K.txt", 95001),
]


def load_ts(path, max_n=0):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = float(line)
                if not np.isnan(v):
                    vals.append(v)
            except ValueError:
                pass
    if max_n > 0:
        vals = vals[:max_n]
    return vals


def bench(func, *args, repeat=REPEAT, **kwargs):
    times = []
    result = None
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
    if time_s is None or time_s <= 0:
        return None
    return (n_subseq * n_subseq / time_s) / 1e6


def benchmark_dataset(name, ts, window, num_threads, use_gpu, repeat):
    n = len(ts)
    n_sub = n - window + 1
    result = {
        'dataset': name,
        'series_length': n,
        'window_size': window,
        'n_subsequences': n_sub,
    }

    print(f"    C22 CPU ({num_threads} threads)...", end=" ", flush=True)
    t_c22_cpu, _ = bench(pyscamp.selfjoin_c22, ts, window,
                         threads=num_threads, gpu=False, repeat=repeat)
    if t_c22_cpu is not None:
        result['c22_cpu'] = {
            'time_s': t_c22_cpu,
            'throughput_mpairs_s': throughput(n_sub, t_c22_cpu),
        }
        print(f"{t_c22_cpu:.3f}s ({throughput(n_sub, t_c22_cpu):.1f} M/s)")
    else:
        result['c22_cpu'] = {'error': 'failed'}
        print("FAILED")

    if use_gpu:
        print(f"    C22 GPU...", end=" ", flush=True)
        t_c22_gpu, _ = bench(pyscamp.selfjoin_c22, ts, window,
                             threads=num_threads, gpu=True, repeat=repeat)
        if t_c22_gpu is not None:
            result['c22_gpu'] = {
                'time_s': t_c22_gpu,
                'throughput_mpairs_s': throughput(n_sub, t_c22_gpu),
            }
            print(f"{t_c22_gpu:.3f}s ({throughput(n_sub, t_c22_gpu):.1f} M/s)")
            if 'time_s' in result.get('c22_cpu', {}):
                result['c22_gpu_speedup'] = result['c22_cpu']['time_s'] / t_c22_gpu
                print(f"    C22 GPU speedup: {result['c22_gpu_speedup']:.2f}x")
        else:
            result['c22_gpu'] = {'error': 'failed'}
            print("FAILED")

    print(f"    MP  CPU ({num_threads} threads)...", end=" ", flush=True)
    t_mp_cpu, _ = bench(pyscamp.selfjoin, ts, window,
                        threads=num_threads, gpus=[], repeat=repeat)
    if t_mp_cpu is not None:
        result['mp_cpu'] = {
            'time_s': t_mp_cpu,
            'throughput_mpairs_s': throughput(n_sub, t_mp_cpu),
        }
        print(f"{t_mp_cpu:.3f}s ({throughput(n_sub, t_mp_cpu):.1f} M/s)")
    else:
        result['mp_cpu'] = {'error': 'failed'}
        print("FAILED")

    if use_gpu:
        print(f"    MP  GPU...", end=" ", flush=True)
        t_mp_gpu, _ = bench(pyscamp.selfjoin, ts, window,
                            gpus=[0], threads=num_threads, repeat=repeat)
        if t_mp_gpu is not None:
            result['mp_gpu'] = {
                'time_s': t_mp_gpu,
                'throughput_mpairs_s': throughput(n_sub, t_mp_gpu),
            }
            print(f"{t_mp_gpu:.3f}s ({throughput(n_sub, t_mp_gpu):.1f} M/s)")
            if 'time_s' in result.get('mp_cpu', {}):
                result['mp_gpu_speedup'] = result['mp_cpu']['time_s'] / t_mp_gpu
                print(f"    MP  GPU speedup: {result['mp_gpu_speedup']:.2f}x")
        else:
            result['mp_gpu'] = {'error': 'failed'}
            print("FAILED")

    return result


def benchmark_thread_scaling(ts, dataset_name, window, thread_counts, use_gpu, repeat):
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
        t, _ = bench(pyscamp.selfjoin_c22, ts, window, threads=tc, gpu=False, repeat=repeat)
        if t is None:
            continue
        if base_time is None:
            base_time = t
        tp = throughput(n_sub, t)
        scaling['thread_counts'].append(tc)
        scaling['c22_times'].append(t)
        scaling['c22_throughputs'].append(tp)
        scaling['c22_speedups'].append(base_time / t)
        print(f"    {tc:>2} threads: {t:.3f}s  {tp:.1f} M/s  {base_time/t:.2f}x")

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
            print(f"{t_gpu:.3f}s")

    return scaling


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark C22 vs MP on SampleInput datasets (SCAMP paper style)')
    parser.add_argument('--gpu', action='store_true', help='Also benchmark GPU')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU only')
    parser.add_argument('--threads', type=int, default=None,
                        help='CPU threads (default: auto)')
    parser.add_argument('--window-size', '-m', type=int, default=DEFAULT_WINDOW,
                        help=f'Window size (default: {DEFAULT_WINDOW})')
    parser.add_argument('--repeat', type=int, default=REPEAT,
                        help=f'Repetitions (default: {REPEAT})')
    parser.add_argument('--max-n', type=int, default=0,
                        help='Max series length to test (0 = no limit)')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'benchmark_sampleinput.json'),
                        help='Output JSON file')
    parser.add_argument('--skip-thread-scaling', action='store_true')
    parser.add_argument('--stdout', action='store_true',
                        help='Print JSON to stdout')
    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu_only and HAS_GPU
    num_threads = args.threads or multiprocessing.cpu_count()

    real_stdout = sys.stdout
    if args.stdout:
        sys.stdout = sys.stderr

    print("=" * 72)
    print("  C22 vs MP — SampleInput Benchmark (SCAMP paper datasets)")
    print("=" * 72)
    print(f"  Window: {args.window_size}  Threads: {num_threads}  "
          f"GPU: {'YES' if use_gpu else 'NO'}  Repeat: {args.repeat}")
    if args.max_n > 0:
        print(f"  Max N: {args.max_n:,}")
    print("=" * 72)

    all_results = {
        'config': {
            'window_size': args.window_size,
            'cpu_threads': num_threads,
            'gpu_enabled': use_gpu,
            'gpu_supported': HAS_GPU,
            'repeat': args.repeat,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'per_dataset': [],
    }

    for name, filename, expected_n in DATASETS:
        filepath = SAMPLE_DIR / filename
        if not filepath.exists():
            print(f"\n  SKIP {name}: {filename} not found")
            continue

        ts = load_ts(str(filepath))
        if args.max_n > 0 and len(ts) > args.max_n:
            print(f"\n  SKIP {name}: N={len(ts):,} > max_n={args.max_n:,}")
            continue

        print(f"\n{'─' * 72}")
        print(f"  {name}  (N={len(ts):,}, w={args.window_size})")
        print(f"{'─' * 72}")

        r = benchmark_dataset(name, ts, args.window_size, num_threads,
                              use_gpu, args.repeat)
        all_results['per_dataset'].append(r)

    if not args.skip_thread_scaling:
        ts_8k = load_ts(str(SAMPLE_DIR / "randomlist8K.txt"))
        tc_list = [t for t in THREAD_SCALING_COUNTS if t <= multiprocessing.cpu_count()]
        all_results['thread_scaling'] = benchmark_thread_scaling(
            ts_8k, "randomlist8K", args.window_size, tc_list, use_gpu, args.repeat)

    scaling = {
        'window_size': args.window_size,
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
            'c22_gpu_times': [], 'c22_gpu_throughputs': [],
            'mp_gpu_times': [], 'mp_gpu_throughputs': [],
        })
    for d in sorted(all_results['per_dataset'], key=lambda x: x['series_length']):
        if 'time_s' not in d.get('c22_cpu', {}) or 'time_s' not in d.get('mp_cpu', {}):
            continue
        scaling['sizes'].append(d['series_length'])
        scaling['dataset_names'].append(d['dataset'])
        scaling['c22_cpu_times'].append(d['c22_cpu']['time_s'])
        scaling['c22_cpu_throughputs'].append(d['c22_cpu']['throughput_mpairs_s'])
        scaling['mp_cpu_times'].append(d['mp_cpu']['time_s'])
        scaling['mp_cpu_throughputs'].append(d['mp_cpu']['throughput_mpairs_s'])
        if use_gpu:
            scaling['c22_gpu_times'].append(d.get('c22_gpu', {}).get('time_s'))
            scaling['c22_gpu_throughputs'].append(d.get('c22_gpu', {}).get('throughput_mpairs_s'))
            scaling['mp_gpu_times'].append(d.get('mp_gpu', {}).get('time_s'))
            scaling['mp_gpu_throughputs'].append(d.get('mp_gpu', {}).get('throughput_mpairs_s'))
    all_results['scaling_by_size'] = scaling

    per_ds = all_results['per_dataset']
    c22_speedups = [d['c22_gpu_speedup'] for d in per_ds if 'c22_gpu_speedup' in d]
    mp_speedups = [d['mp_gpu_speedup'] for d in per_ds if 'mp_gpu_speedup' in d]
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
    all_results['summary'] = summary
    all_results['config']['n_datasets'] = len(per_ds)

    json_str = json.dumps(all_results, indent=2, default=lambda o: None)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(json_str)

    if args.stdout:
        print(json_str, file=real_stdout)

    print(f"\n{'=' * 72}")
    print("  BENCHMARK COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Datasets: {len(per_ds)}")
    if c22_speedups:
        print(f"  C22 GPU speedup: mean={np.mean(c22_speedups):.2f}x, "
              f"range=[{np.min(c22_speedups):.2f}x, {np.max(c22_speedups):.2f}x]")
    if mp_speedups:
        print(f"  MP  GPU speedup: mean={np.mean(mp_speedups):.2f}x, "
              f"range=[{np.min(mp_speedups):.2f}x, {np.max(mp_speedups):.2f}x]")
    print(f"\n  Results: {output_path}")
    print(f"  Plot:    python3 test/plot_20papers_results.py -i {output_path}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
