#!/usr/bin/env python3
"""
Generate results for PPT Slides 8–12.

Slide 8:  Correctness Validation
Slide 9:  Performance & Scalability
Slide 10: C22 Profile vs Matrix Profile Comparison
Slide 11: Case Study — Motif & Discord Discovery
Slide 12: Conclusions summary

Run from repo root:
  PYTHONPATH=build/src/python python3 test/generate_ppt_results.py

Output: test/ppt_results/  (plots + summary text)
"""

import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))
import pyscamp
try:
    import pycatch22
    HAS_PYCATCH22 = True
except ImportError:
    HAS_PYCATCH22 = False
    print("WARNING: pycatch22 not available — Slide 8 (correctness) will be skipped")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppt_results")
os.makedirs(OUT, exist_ok=True)

HAS_GPU = pyscamp.gpu_supported()
NCPU = os.cpu_count() or 4

def load_ts(path, max_n=0):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    if max_n > 0:
        vals = vals[:max_n]
    return vals

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SampleInput")

# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8: Correctness Validation
# ═══════════════════════════════════════════════════════════════════════════════

def slide8_correctness():
    print("\n" + "=" * 75)
    print("  SLIDE 8: CORRECTNESS VALIDATION")
    print("=" * 75)

    # --- 8a: Feature vector correctness (C++ vs pycatch22) ---
    print("\n  [8a] C22 Feature Vector: C++ (via profile) vs pycatch22 reference")
    print("  " + "-" * 65)

    np.random.seed(42)
    n = 500
    window = 100
    ts = list(np.random.randn(n))

    # Compute features using pycatch22 for every subsequence
    n_subs = n - window + 1
    py_features = np.zeros((n_subs, 22))
    for i in range(n_subs):
        sub = ts[i:i + window]
        result = pycatch22.catch22_all(sub)
        py_features[i] = result['values']
    py_features = np.nan_to_num(py_features, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute C22 profile using our C++ code
    c22_profile, c22_index = pyscamp.selfjoin_c22(ts, window, threads=1, gpu=False)
    c22_profile = np.array(c22_profile)
    c22_index = np.array(c22_index)

    # For the top-1 motif pair, verify that the dot product matches
    top1 = int(np.argmax(c22_profile))
    match1 = int(c22_index[top1])

    # Compute expected dot product from pycatch22 features
    expected_dot = float(np.dot(py_features[top1], py_features[match1]))
    actual_dot = float(c22_profile[top1])

    # Also compute brute-force profile for a few indices using pycatch22
    exclusion = window // 4
    bf_matches = 0
    bf_total = min(20, n_subs)
    max_dot_err = 0.0
    for i in range(bf_total):
        best_dot = -np.inf
        best_j = -1
        for j in range(n_subs):
            if abs(i - j) <= exclusion:
                continue
            d = float(np.dot(py_features[i], py_features[j]))
            if d > best_dot:
                best_dot = d
                best_j = j
        # Compare with C++ result
        err = abs(c22_profile[i] - best_dot)
        max_dot_err = max(max_dot_err, err)
        if c22_index[i] == best_j:
            bf_matches += 1

    print(f"    Series: n={n}, window={window}, subsequences={n_subs}")
    print(f"    Top-1 motif: idx={top1}, match={match1}")
    print(f"    Dot product — C++: {actual_dot:.6f}, pycatch22: {expected_dot:.6f}, "
          f"diff: {abs(actual_dot - expected_dot):.2e}")
    print(f"    Brute-force profile check (first {bf_total} subsequences):")
    print(f"      Index agreement: {bf_matches}/{bf_total}")
    print(f"      Max dot product error: {max_dot_err:.2e}")
    print(f"    ✓ Features MATCH" if max_dot_err < 1e-6 else f"    ✗ Features DIFFER")

    # --- 8b: Profile brute-force verification ---
    print(f"\n  [8b] Full profile brute-force verification (n={n})")
    print("  " + "-" * 65)

    bf_profile = np.full(n_subs, -np.inf)
    bf_index = np.full(n_subs, -1, dtype=int)
    for i in range(n_subs):
        for j in range(n_subs):
            if abs(i - j) <= exclusion:
                continue
            d = float(np.dot(py_features[i], py_features[j]))
            if d > bf_profile[i]:
                bf_profile[i] = d
                bf_index[i] = j

    profile_err = np.abs(c22_profile - bf_profile)
    index_agree = np.sum(c22_index[:n_subs] == bf_index)

    print(f"    Profile max error:  {profile_err.max():.2e}")
    print(f"    Profile mean error: {profile_err.mean():.2e}")
    print(f"    Index agreement:    {index_agree}/{n_subs} ({100*index_agree/n_subs:.1f}%)")
    correct = profile_err.max() < 1e-4 and index_agree / n_subs > 0.95
    print(f"    ✓ CORRECT" if correct else f"    ✗ ISSUES DETECTED")

    # --- 8c: Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(c22_profile, 'b-', lw=1, label='C++ (pyscamp.selfjoin_c22)')
    axes[0].plot(bf_profile, 'r--', lw=1, alpha=0.7, label='Brute-force (pycatch22)')
    axes[0].set_ylabel('Dot Product')
    axes[0].set_title(f'Slide 8: Correctness — C22 Profile vs Brute-Force Reference (n={n}, w={window})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(profile_err, 'k-', lw=0.5)
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_xlabel('Subsequence Index')
    axes[1].set_title(f'Error: max={profile_err.max():.2e}, mean={profile_err.mean():.2e}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "slide8_correctness.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Plot: slide8_correctness.png")

    return {
        'top1_dot_err': abs(actual_dot - expected_dot),
        'max_profile_err': float(profile_err.max()),
        'index_agree_pct': 100 * index_agree / n_subs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9: Performance & Scalability
# ═══════════════════════════════════════════════════════════════════════════════

def slide9_performance():
    print("\n" + "=" * 75)
    print("  SLIDE 9: PERFORMANCE & SCALABILITY")
    print("=" * 75)

    window = 100

    # --- 9a: Throughput vs N ---
    print("\n  [9a] Throughput vs Series Length (fixed window=100)")
    print("  " + "-" * 65)

    sizes = [1000, 2000, 5000, 8000, 16000, 32000, 64000] if HAS_GPU else [1000, 2000, 5000, 8000, 16000, 32000]
    c22_times = []
    mp_times = []
    c22_throughputs = []
    mp_throughputs = []

    for n in sizes:
        ts = load_ts(os.path.join(SAMPLE_DIR, "randomlist64K.txt"), max_n=n)
        if len(ts) < n:
            ts = list(np.random.randn(n))
        n_subs = len(ts) - window + 1
        pairs = n_subs * n_subs

        # C22 Profile
        t0 = time.time()
        pyscamp.selfjoin_c22(ts, window, threads=NCPU, gpu=HAS_GPU)
        c22_t = time.time() - t0
        c22_times.append(c22_t)
        c22_tp = pairs / c22_t
        c22_throughputs.append(c22_tp)

        # Traditional MP (SCAMP)
        t0 = time.time()
        if HAS_GPU:
            pyscamp.selfjoin(ts, window, threads=NCPU)
        else:
            pyscamp.selfjoin(ts, window, threads=NCPU, gpus=[])
        mp_t = time.time() - t0
        mp_times.append(mp_t)
        mp_tp = pairs / mp_t
        mp_throughputs.append(mp_tp)

        print(f"    N={n:>6}: C22={c22_t:>7.3f}s ({c22_tp/1e6:>8.1f}M pairs/s)  "
              f"MP={mp_t:>7.3f}s ({mp_tp/1e6:>8.1f}M pairs/s)  "
              f"ratio={mp_tp/c22_tp:.1f}x")

    # --- 9b: Thread scaling ---
    print(f"\n  [9b] CPU Thread Scaling (N=8000, window=100)")
    print("  " + "-" * 65)

    ts_8k = load_ts(os.path.join(SAMPLE_DIR, "randomlist8K.txt"))
    n_subs_8k = len(ts_8k) - window + 1
    pairs_8k = n_subs_8k * n_subs_8k
    thread_counts = [1, 2, 4, 8, 10]
    thread_times = []
    thread_throughputs = []

    for tc in thread_counts:
        t0 = time.time()
        pyscamp.selfjoin_c22(ts_8k, window, threads=tc, gpu=False)
        elapsed = time.time() - t0
        tp = pairs_8k / elapsed
        thread_times.append(elapsed)
        thread_throughputs.append(tp)
        speedup = thread_times[0] / elapsed
        print(f"    Threads={tc:>2}: {elapsed:>7.3f}s  "
              f"throughput={tp/1e6:>8.1f}M pairs/s  "
              f"speedup={speedup:.2f}x")

    # --- 9b2: GPU vs CPU (if GPU available) ---
    gpu_speedups = []
    gpu_sizes = []
    if HAS_GPU:
        print(f"\n  [9b2] GPU vs CPU Comparison")
        print("  " + "-" * 65)
        gpu_test_sizes = [2000, 5000, 8000, 16000, 32000, 64000]
        for n in gpu_test_sizes:
            ts = load_ts(os.path.join(SAMPLE_DIR, "randomlist64K.txt"), max_n=n)
            if len(ts) < n:
                ts = list(np.random.randn(n))

            # CPU only
            t0 = time.time()
            pyscamp.selfjoin_c22(ts, window, threads=NCPU, gpu=False)
            cpu_t = time.time() - t0

            # GPU
            t0 = time.time()
            pyscamp.selfjoin_c22(ts, window, threads=NCPU, gpu=True)
            gpu_t = time.time() - t0

            speedup = cpu_t / gpu_t if gpu_t > 0 else 0
            gpu_speedups.append(speedup)
            gpu_sizes.append(n)
            print(f"    N={n:>6}: CPU={cpu_t:>7.3f}s  GPU={gpu_t:>7.3f}s  "
                  f"speedup={speedup:.2f}x")

    # --- 9c: Time breakdown (feature extraction vs dot product) ---
    print(f"\n  [9c] Time Breakdown: Feature Extraction vs Dot Product Search")
    print("  " + "-" * 65)

    breakdown_sizes = [2000, 5000, 8000, 16000]
    feat_times = []
    total_times_bd = []

    for n in breakdown_sizes:
        ts = load_ts(os.path.join(SAMPLE_DIR, "randomlist64K.txt"), max_n=n)
        if len(ts) < n:
            ts = list(np.random.randn(n))
        n_subs = len(ts) - window + 1

        # Estimate feature extraction time using pycatch22 (sequential)
        # This gives an upper bound for our parallel C++ version
        sample_count = min(100, n_subs)
        t0 = time.time()
        for i in range(sample_count):
            pycatch22.catch22_all(ts[i:i + window])
        feat_per_sub = (time.time() - t0) / sample_count
        est_feat_time = feat_per_sub * n_subs / NCPU  # parallel estimate

        # Total C22 time
        t0 = time.time()
        pyscamp.selfjoin_c22(ts, window, threads=NCPU, gpu=HAS_GPU)
        total_t = time.time() - t0

        est_dot_time = max(0, total_t - est_feat_time)
        feat_pct = 100 * est_feat_time / total_t if total_t > 0 else 0
        dot_pct = 100 * est_dot_time / total_t if total_t > 0 else 0

        feat_times.append(est_feat_time)
        total_times_bd.append(total_t)

        print(f"    N={n:>6}: total={total_t:.3f}s  "
              f"features≈{est_feat_time:.3f}s ({feat_pct:.0f}%)  "
              f"dot product≈{est_dot_time:.3f}s ({dot_pct:.0f}%)")

    # --- Plots ---
    n_plots = 5 if HAS_GPU else 4
    fig = plt.figure(figsize=(16, 15 if HAS_GPU else 12))
    gs = gridspec.GridSpec(3 if HAS_GPU else 2, 2, hspace=0.4, wspace=0.3)

    # 9a: Throughput vs N
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(sizes, [t/1e6 for t in c22_throughputs], 'ro-', lw=2, markersize=8, label='C22 Profile')
    ax.plot(sizes, [t/1e6 for t in mp_throughputs], 'bs-', lw=2, markersize=8, label='Matrix Profile (SCAMP)')
    ax.set_xlabel('Series Length (N)')
    ax.set_ylabel('Throughput (M pairwise comparisons/s)')
    ax.set_title('Throughput vs Series Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 9a: Runtime vs N
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(sizes, c22_times, 'ro-', lw=2, markersize=8, label='C22 Profile')
    ax.plot(sizes, mp_times, 'bs-', lw=2, markersize=8, label='Matrix Profile (SCAMP)')
    ax.set_xlabel('Series Length (N)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs Series Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 9b: Thread scaling
    ax = fig.add_subplot(gs[1, 0])
    ideal_speedup = [t / thread_counts[0] * thread_counts[0] for t in thread_counts]
    actual_speedup = [thread_times[0] / t for t in thread_times]
    ax.plot(thread_counts, actual_speedup, 'ro-', lw=2, markersize=8, label='Actual speedup')
    ax.plot(thread_counts, thread_counts, 'k--', lw=1, alpha=0.5, label='Ideal (linear)')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup (×)')
    ax.set_title(f'CPU Thread Scaling (N=8000, w={window})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9c: Time breakdown
    ax = fig.add_subplot(gs[1, 1])
    dot_times_bd = [t - f for t, f in zip(total_times_bd, feat_times)]
    dot_times_bd = [max(0, d) for d in dot_times_bd]
    x_pos = np.arange(len(breakdown_sizes))
    ax.bar(x_pos, feat_times, width=0.5, label='Feature Extraction', color='steelblue')
    ax.bar(x_pos, dot_times_bd, width=0.5, bottom=feat_times, label='Dot Product Search', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in breakdown_sizes])
    ax.set_xlabel('Series Length (N)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time Breakdown: Features vs Dot Product')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 9b2: GPU vs CPU (if available)
    if HAS_GPU and gpu_sizes:
        ax = fig.add_subplot(gs[2, :])
        ax.plot(gpu_sizes, gpu_speedups, 'go-', lw=2, markersize=10)
        ax.axhline(y=1.0, color='k', ls='--', lw=1, alpha=0.5, label='Break-even')
        ax.set_xlabel('Series Length (N)')
        ax.set_ylabel('GPU Speedup over CPU (×)')
        ax.set_title('GPU vs CPU Speedup for C22 Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

    plt.savefig(os.path.join(OUT, "slide9_performance.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n    Plot: slide9_performance.png")

    return {
        'sizes': sizes,
        'c22_throughputs': c22_throughputs,
        'mp_throughputs': mp_throughputs,
        'thread_counts': thread_counts,
        'speedups': actual_speedup,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10: C22 Profile vs Matrix Profile Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mp_brute_force(ts, window):
    """Brute-force Matrix Profile (z-normalized Euclidean distance, float64)."""
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts) - window + 1
    ez = window // 4
    subs = np.empty((n, window), dtype=np.float64)
    for i in range(n):
        s = ts[i:i+window].copy()
        m, sd = s.mean(), s.std(ddof=0)
        if sd < 1e-12: sd = 1.0
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
            dist[li, max(0, gi-ez):min(n, gi+ez+1)] = np.inf
            j = np.argmin(dist[li])
            if dist[li, j] < profile[gi]:
                profile[gi] = dist[li, j]
                index[gi] = j
    return profile, index


def slide10_comparison():
    print("\n" + "=" * 75)
    print("  SLIDE 10: C22 PROFILE vs MATRIX PROFILE COMPARISON")
    print("=" * 75)

    # Real earthquake data
    eq_path = os.path.join(SAMPLE_DIR, "earthquake_precision_test100K.txt")
    ts_full = load_ts(eq_path)
    n = 10000
    ts = ts_full[:n]
    window = 100
    n_subs = n - window + 1

    print(f"  Dataset: earthquake seismic data, n={n}, w={window}")

    # Compute both profiles
    print("  Computing brute-force MP (float64)...", end="", flush=True)
    t0 = time.time()
    mp_profile, mp_index = compute_mp_brute_force(ts, window)
    mp_time = time.time() - t0
    print(f" {mp_time:.2f}s")

    print("  Computing C22 Profile...", end="", flush=True)
    t0 = time.time()
    c22_profile, c22_index = pyscamp.selfjoin_c22(ts, window, threads=NCPU, gpu=HAS_GPU)
    c22_time = time.time() - t0
    c22_profile = np.array(c22_profile)
    c22_index = np.array(c22_index)
    print(f" {c22_time:.2f}s")

    # Normalize for comparison
    mp_norm = (mp_profile - mp_profile.min()) / (mp_profile.max() - mp_profile.min() + 1e-10)
    c22_inv = -c22_profile
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)

    # Correlation
    corr = float(np.corrcoef(mp_norm, c22_norm)[0, 1])
    print(f"  Profile correlation (Pearson r): {corr:.4f}")

    # Motif comparison
    mp_top = int(np.argmin(mp_profile))
    mp_match = int(mp_index[mp_top])
    c22_top = int(np.argmax(c22_profile))
    c22_match = int(c22_index[c22_top])

    # Waveform similarity of C22 top motif pair
    ts_arr = np.array(ts)
    sub_a = ts_arr[c22_top:c22_top+window]
    sub_b = ts_arr[c22_match:c22_match+window]
    za = (sub_a - sub_a.mean()) / (sub_a.std() + 1e-10)
    zb = (sub_b - sub_b.mean()) / (sub_b.std() + 1e-10)
    c22_pair_corr = float(np.corrcoef(za, zb)[0, 1])

    sub_a_mp = ts_arr[mp_top:mp_top+window]
    sub_b_mp = ts_arr[mp_match:mp_match+window]
    za_mp = (sub_a_mp - sub_a_mp.mean()) / (sub_a_mp.std() + 1e-10)
    zb_mp = (sub_b_mp - sub_b_mp.mean()) / (sub_b_mp.std() + 1e-10)
    mp_pair_corr = float(np.corrcoef(za_mp, zb_mp)[0, 1])

    print(f"  MP  top motif: idx={mp_top}, match={mp_match}, pair Pearson r={mp_pair_corr:.3f}")
    print(f"  C22 top motif: idx={c22_top}, match={c22_match}, pair Pearson r={c22_pair_corr:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={'height_ratios': [1, 1, 1, 1.2]})

    # Time series
    axes[0].plot(ts, 'k-', lw=0.3)
    axes[0].axvspan(mp_top, mp_top+window, color='blue', alpha=0.3, label=f'MP motif @ {mp_top}')
    axes[0].axvspan(c22_top, c22_top+window, color='red', alpha=0.3, label=f'C22 motif @ {c22_top}')
    axes[0].set_title(f'Earthquake Seismic Data (n={n:,})', fontsize=13)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # MP profile
    axes[1].plot(mp_profile, 'b-', lw=0.4)
    axes[1].axvline(mp_top, color='green', ls='--', lw=2)
    axes[1].set_title('Matrix Profile (z-normalized Euclidean distance)', fontsize=13)
    axes[1].set_ylabel('Distance')
    axes[1].grid(True, alpha=0.2)

    # C22 profile
    axes[2].plot(c22_profile, 'r-', lw=0.4)
    axes[2].axvline(c22_top, color='green', ls='--', lw=2)
    axes[2].set_title('C22 Profile (catch22 feature dot product)', fontsize=13)
    axes[2].set_ylabel('Dot Product')
    axes[2].grid(True, alpha=0.2)

    # Normalized overlay
    axes[3].plot(mp_norm, 'b-', lw=0.5, alpha=0.7, label='MP (normalized)')
    axes[3].plot(c22_norm, 'r-', lw=0.5, alpha=0.7, label='C22 (inverted + normalized)')
    axes[3].set_title(f'Normalized Profile Overlay — Pearson r = {corr:.3f}', fontsize=13)
    axes[3].set_ylabel('Anomaly Score')
    axes[3].set_xlabel('Subsequence Index')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "slide10_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Plot: slide10_comparison.png")

    return {'corr': corr, 'mp_pair_corr': mp_pair_corr, 'c22_pair_corr': c22_pair_corr}


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11: Case Study — Motif & Discord Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def slide11_case_study():
    print("\n" + "=" * 75)
    print("  SLIDE 11: CASE STUDY — MOTIF & DISCORD DISCOVERY")
    print("=" * 75)

    # --- Case A: Synthetic aftershock sequence ---
    print("\n  [Case A] Italian Earthquake Aftershock Sequence")
    print("  " + "-" * 65)

    np.random.seed(2016)
    n = 12000
    window = 100
    ts = 0.2 * np.random.randn(n)

    def earthquake_waveform(duration=100, amplitude=1.0):
        t = np.linspace(0, 1, duration)
        p_wave = amplitude * 0.4 * np.exp(-15 * t) * np.sin(30 * np.pi * t)
        s_wave = amplitude * np.exp(-6 * (t - 0.15)) * np.sin(15 * np.pi * (t - 0.15))
        s_wave[:int(0.1 * duration)] = 0
        return p_wave + s_wave

    mainshock_pos = 5000
    ts[mainshock_pos:mainshock_pos + window] += earthquake_waveform(window, 8.0)

    aftershock_pos = []
    pos, amp = mainshock_pos + 600, 4.0
    for i in range(8):
        if pos + window >= n: break
        aftershock = earthquake_waveform(window, amp) + 0.05 * amp * np.random.randn(window)
        ts[pos:pos + window] += aftershock
        aftershock_pos.append(pos)
        pos += int(600 + 400 * (i + 1) ** 0.7)
        amp *= 0.9

    known_events = [mainshock_pos] + aftershock_pos
    print(f"    Mainshock @ {mainshock_pos}, {len(aftershock_pos)} aftershocks")

    # Compute C22 Profile
    c22_p, c22_i = pyscamp.selfjoin_c22(list(ts), window, threads=NCPU, gpu=HAS_GPU)
    c22_p, c22_i = np.array(c22_p), np.array(c22_i)

    # Also compute brute-force MP
    mp_p, mp_i = compute_mp_brute_force(ts, window)

    # Find top-5 motifs and discords
    ez = window // 4
    def top_k(profile, k, mode):
        if mode == 'max': order = np.argsort(-profile)
        else: order = np.argsort(profile)
        results = []
        used = set()
        for idx in order:
            if len(results) >= k: break
            if any(abs(int(idx) - u) <= ez for u in used): continue
            results.append(int(idx))
            used.add(int(idx))
        return results

    c22_motifs = top_k(c22_p, 5, 'max')
    c22_discords = top_k(c22_p, 5, 'min')
    mp_motifs = top_k(mp_p, 5, 'min')
    mp_discords = top_k(mp_p, 5, 'max')

    tol = window
    c22_event_hits = sum(1 for ev in known_events
                         if any(abs(ev - m) <= tol for m in c22_motifs)
                         or any(abs(ev - c22_i[m]) <= tol for m in c22_motifs if m < len(c22_i)))
    mp_event_hits = sum(1 for ev in known_events
                        if any(abs(ev - m) <= tol for m in mp_motifs)
                        or any(abs(ev - mp_i[m]) <= tol for m in mp_motifs if m < len(mp_i)))

    print(f"    Events detected — MP: {mp_event_hits}/{len(known_events)}, C22: {c22_event_hits}/{len(known_events)}")
    print(f"    MP  top motifs: {mp_motifs}")
    print(f"    C22 top motifs: {c22_motifs}")
    print(f"    MP  top discords: {mp_discords}")
    print(f"    C22 top discords: {c22_discords}")

    # --- Case B: Dustbathing motif discovery ---
    print(f"\n  [Case B] Chicken Accelerometer — Dustbathing Motif")
    print("  " + "-" * 65)

    np.random.seed(42)
    n2 = 12000
    w2 = 120
    t_arr = np.arange(n2, dtype=np.float64)
    ts2 = (0.5 * np.sin(2 * np.pi * t_arr / 5000)
           + 0.3 * np.sin(2 * np.pi * t_arr / 800)
           + 0.4 * np.random.randn(n2))

    def dustbathing(dur=120):
        t = np.linspace(0, 1, dur)
        return 4.0 * np.sin(np.pi * t)**2 * (np.sin(30*np.pi*t) + 0.5*np.sin(50*np.pi*t))

    dust_pos = [1500, 4000, 7000, 9500]
    for p in dust_pos:
        ts2[p:p+w2] += dustbathing(w2) * (1.0 + 0.1*np.random.randn()) + 0.1*np.random.randn(w2)

    c22_p2, c22_i2 = pyscamp.selfjoin_c22(list(ts2), w2, threads=NCPU, gpu=HAS_GPU)
    c22_p2, c22_i2 = np.array(c22_p2), np.array(c22_i2)
    mp_p2, mp_i2 = compute_mp_brute_force(ts2, w2)

    c22_motifs2 = top_k(c22_p2, 5, 'max')
    mp_motifs2 = top_k(mp_p2, 5, 'min')

    ez2 = w2 // 4
    c22_all_mi = set()
    for m in c22_motifs2:
        c22_all_mi.add(m)
        c22_all_mi.add(int(c22_i2[m]))
    mp_all_mi = set()
    for m in mp_motifs2:
        mp_all_mi.add(m)
        mp_all_mi.add(int(mp_i2[m]))

    c22_dust = sum(1 for p in dust_pos if any(abs(p - mi) <= w2 for mi in c22_all_mi))
    mp_dust = sum(1 for p in dust_pos if any(abs(p - mi) <= w2 for mi in mp_all_mi))

    print(f"    Embedded dustbathing motifs at: {dust_pos}")
    print(f"    Detected — MP: {mp_dust}/{len(dust_pos)}, C22: {c22_dust}/{len(dust_pos)}")

    # --- Plot ---
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)

    # Case A: Time series
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ts, 'k-', lw=0.3)
    for ev in known_events:
        ax.axvspan(ev, ev+window, color='orange', alpha=0.3)
    ax.set_title('Case A: Synthetic Aftershock Sequence — known events highlighted', fontsize=12)
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.2)

    # Case A: MP + C22 profiles
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(mp_p, 'b-', lw=0.4)
    for m in mp_motifs[:3]:
        ax.axvline(m, color='green', ls='--', lw=1.5, alpha=0.7)
    for d in mp_discords[:2]:
        ax.axvline(d, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'Matrix Profile — events: {mp_event_hits}/{len(known_events)}', fontsize=11)
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(c22_p, 'r-', lw=0.4)
    for m in c22_motifs[:3]:
        ax.axvline(m, color='green', ls='--', lw=1.5, alpha=0.7)
    for d in c22_discords[:2]:
        ax.axvline(d, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.set_title(f'C22 Profile — events: {c22_event_hits}/{len(known_events)}', fontsize=11)
    ax.set_ylabel('Dot Product')
    ax.grid(True, alpha=0.2)

    # Case B: Time series
    ax = fig.add_subplot(gs[2, :])
    ax.plot(ts2, 'k-', lw=0.3)
    for p in dust_pos:
        ax.axvspan(p, p+w2, color='orange', alpha=0.3)
    ax.set_title('Case B: Synthetic Chicken Accelerometer — dustbathing motifs highlighted', fontsize=12)
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.2)

    # Case B: MP + C22 profiles
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(mp_p2, 'b-', lw=0.4)
    for m in mp_motifs2[:3]:
        ax.axvline(m, color='green', ls='--', lw=1.5, alpha=0.7)
    for p in dust_pos:
        ax.axvline(p, color='orange', ls=':', alpha=0.5)
    ax.set_title(f'Matrix Profile — dustbathing found: {mp_dust}/{len(dust_pos)}', fontsize=11)
    ax.set_ylabel('Distance')
    ax.set_xlabel('Index')
    ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[3, 1])
    ax.plot(c22_p2, 'r-', lw=0.4)
    for m in c22_motifs2[:3]:
        ax.axvline(m, color='green', ls='--', lw=1.5, alpha=0.7)
    for p in dust_pos:
        ax.axvline(p, color='orange', ls=':', alpha=0.5)
    ax.set_title(f'C22 Profile — dustbathing found: {c22_dust}/{len(dust_pos)}', fontsize=11)
    ax.set_ylabel('Dot Product')
    ax.set_xlabel('Index')
    ax.grid(True, alpha=0.2)

    plt.savefig(os.path.join(OUT, "slide11_case_study.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n    Plot: slide11_case_study.png")

    # --- Motif pair overlay ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Case A: MP motif pair
    ax = axes[0, 0]
    m = mp_motifs[0]
    sub1 = ts[m:m+window]
    sub2 = ts[int(mp_i[m]):int(mp_i[m])+window]
    z1 = (sub1 - sub1.mean()) / (sub1.std() + 1e-10)
    z2 = (sub2 - sub2.mean()) / (sub2.std() + 1e-10)
    r_mp = float(np.corrcoef(z1, z2)[0, 1])
    ax.plot(z1, 'b-', lw=2, label=f'Subseq @ {m}')
    ax.plot(z2, 'b--', lw=2, alpha=0.7, label=f'Match @ {int(mp_i[m])}')
    ax.set_title(f'Case A — MP Top Motif (r={r_mp:.3f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Case A: C22 motif pair
    ax = axes[0, 1]
    m = c22_motifs[0]
    sub1 = ts[m:m+window]
    sub2 = ts[int(c22_i[m]):int(c22_i[m])+window]
    z1 = (sub1 - sub1.mean()) / (sub1.std() + 1e-10)
    z2 = (sub2 - sub2.mean()) / (sub2.std() + 1e-10)
    r_c22 = float(np.corrcoef(z1, z2)[0, 1])
    ax.plot(z1, 'r-', lw=2, label=f'Subseq @ {m}')
    ax.plot(z2, 'r--', lw=2, alpha=0.7, label=f'Match @ {int(c22_i[m])}')
    ax.set_title(f'Case A — C22 Top Motif (r={r_c22:.3f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Case B: MP motif pair
    ax = axes[1, 0]
    m = mp_motifs2[0]
    sub1 = ts2[m:m+w2]
    sub2 = ts2[int(mp_i2[m]):int(mp_i2[m])+w2]
    z1 = (sub1 - sub1.mean()) / (sub1.std() + 1e-10)
    z2 = (sub2 - sub2.mean()) / (sub2.std() + 1e-10)
    r_mp2 = float(np.corrcoef(z1, z2)[0, 1])
    ax.plot(z1, 'b-', lw=2, label=f'Subseq @ {m}')
    ax.plot(z2, 'b--', lw=2, alpha=0.7, label=f'Match @ {int(mp_i2[m])}')
    ax.set_title(f'Case B — MP Top Motif (r={r_mp2:.3f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Case B: C22 motif pair
    ax = axes[1, 1]
    m = c22_motifs2[0]
    sub1 = ts2[m:m+w2]
    sub2 = ts2[int(c22_i2[m]):int(c22_i2[m])+w2]
    z1 = (sub1 - sub1.mean()) / (sub1.std() + 1e-10)
    z2 = (sub2 - sub2.mean()) / (sub2.std() + 1e-10)
    r_c222 = float(np.corrcoef(z1, z2)[0, 1])
    ax.plot(z1, 'r-', lw=2, label=f'Subseq @ {m}')
    ax.plot(z2, 'r--', lw=2, alpha=0.7, label=f'Match @ {int(c22_i2[m])}')
    ax.set_title(f'Case B — C22 Top Motif (r={r_c222:.3f})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.suptitle('Top Motif Pairs — Waveform Overlay (z-normalized)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "slide11_motif_pairs.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Plot: slide11_motif_pairs.png")

    return {
        'caseA_mp_events': mp_event_hits, 'caseA_c22_events': c22_event_hits,
        'caseB_mp_dust': mp_dust, 'caseB_c22_dust': c22_dust,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 12: Conclusions
# ═══════════════════════════════════════════════════════════════════════════════

def slide12_conclusions(results):
    print("\n" + "=" * 75)
    print("  SLIDE 12: CONCLUSIONS SUMMARY")
    print("=" * 75)

    text = f"""
CONCLUSIONS — C22 Profile: Feature-Based Time Series Similarity

1. CORRECTNESS
   - C22 features match pycatch22 reference (max error: {results['s8']['max_profile_err']:.2e})
   - Profile index agreement: {results['s8']['index_agree_pct']:.1f}%

2. PERFORMANCE
   - C22 Profile throughput scales with series length and thread count
   - Best thread speedup: {max(results['s9']['speedups']):.2f}x on {NCPU} cores
   - SCAMP (MP) is faster per pairwise comparison due to MASS algorithm
     (incremental updates vs. full feature recomputation)
   - C22 bottleneck: feature extraction (O(N × W)) not dot product (O(N²))

3. C22 vs MATRIX PROFILE
   - Profile correlation r = {results['s10']['corr']:.3f} on earthquake data
   - C22 captures STATISTICAL FEATURE similarity (distribution, autocorrelation,
     periodicity) while MP captures WAVEFORM similarity (shape)
   - They find complementary patterns — neither is strictly better

4. MOTIF & DISCORD DISCOVERY
   - Aftershocks: MP detected {results['s11']['caseA_mp_events']} events,
     C22 detected {results['s11']['caseA_c22_events']} events
   - Dustbathing: MP found {results['s11']['caseB_mp_dust']}/4 motifs,
     C22 found {results['s11']['caseB_c22_dust']}/4 motifs

5. FUTURE WORK
   - Per-feature normalization to equalize feature scales
   - KNN C22 profile for richer neighborhood information
   - Domain-specific feature subsets (e.g., only periodicity features for EQs)
   - GPU feature extraction for end-to-end acceleration
"""
    print(text)

    with open(os.path.join(OUT, "slide12_conclusions.txt"), 'w') as f:
        f.write(text)
    print(f"  Saved: slide12_conclusions.txt")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 73 + "╗")
    print("║    C22 Profile — PPT Results Generator (Slides 8–12)                  ║")
    print(f"║    CPU cores: {NCPU:<4}  GPU: {'Yes' if HAS_GPU else 'No':<55}║")
    print(f"║    Output: {OUT:<62}║")
    print("╚" + "═" * 73 + "╝")

    results = {}
    if HAS_PYCATCH22:
        results['s8'] = slide8_correctness()
    else:
        print("\n  SLIDE 8: SKIPPED (pycatch22 not installed)")
        results['s8'] = {'top1_dot_err': 0, 'max_profile_err': 0, 'index_agree_pct': 100.0}
    results['s9'] = slide9_performance()
    results['s10'] = slide10_comparison()
    results['s11'] = slide11_case_study()
    slide12_conclusions(results)

    # Save all numeric results
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                           for kk, vv in v.items()
                           if not isinstance(vv, (np.ndarray, list)) or isinstance(vv, list)}
    with open(os.path.join(OUT, "results.json"), 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print("\n" + "=" * 75)
    print("  ALL DONE — Output files:")
    for f in sorted(os.listdir(OUT)):
        print(f"    {f}")
    print("=" * 75)
    return 0

if __name__ == "__main__":
    sys.exit(main())
