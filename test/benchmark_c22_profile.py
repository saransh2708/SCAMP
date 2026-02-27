#!/usr/bin/env python3
"""
C22 Profile Benchmarking Script
================================
Generates experimental performance charts for the C22 Profile implementation.

Measures:
  1. Scalability with time series length (N)
  2. Scalability with window size (m)
  3. Thread scaling (1, 2, 4, 8, ... up to hardware concurrency)
  4. Feature computation vs dot-product search breakdown

Usage:
  PYTHONPATH=build/src/python python3 test/benchmark_c22_profile.py

Output:
  - test/charts/c22_scaling_by_length.png
  - test/charts/c22_scaling_by_window.png
  - test/charts/c22_thread_scaling.png
  - test/charts/c22_time_breakdown.png
  - test/charts/c22_benchmark_summary.png  (combined)
  - test/charts/benchmark_results.csv       (raw data)
"""

import sys
import os
import time
import csv
import multiprocessing

import numpy as np

# Attempt to import matplotlib; if unavailable, output CSV only
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found — CSV data will be saved but no charts.")

# pyscamp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))
try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Build first, then run with PYTHONPATH=build/src/python")
    sys.exit(1)

HAS_GPU = pyscamp.gpu_supported()

# ── Output directory ──────────────────────────────────────────────────────────
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
HW_THREADS = multiprocessing.cpu_count()
REPEAT = 3  # number of repetitions per measurement (take min)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ts(n, seed=42):
    """Generate a random-walk time series of length n."""
    rng = np.random.default_rng(seed)
    return list(rng.standard_normal(n).cumsum())


def bench(func, *args, repeat=REPEAT, **kwargs):
    """Run func(*args, **kwargs) `repeat` times; return min elapsed time."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)

# ── All results stored here ───────────────────────────────────────────────────
all_results = []

# ============================================================================
# Benchmark 1: Scalability with time series length
# ============================================================================
print("=" * 70)
print("BENCHMARK 1: C22 Profile time vs time series length")
print("=" * 70)

lengths = [500, 1000, 2000, 4000, 8000, 16000]
window_fixed = 100
b1_cpu = []
b1_gpu = []

for n in lengths:
    ts = make_ts(n)
    t_cpu = bench(pyscamp.selfjoin_c22, ts, window_fixed, threads=HW_THREADS, gpu=False)
    b1_cpu.append(t_cpu)
    row = {'benchmark': 'scaling_by_length', 'N': n, 'window': window_fixed,
           'threads': HW_THREADS, 'device': 'CPU', 'time_s': t_cpu}
    all_results.append(row)
    print(f"  N={n:>6}, CPU: {t_cpu:.3f}s")

    if HAS_GPU:
        t_gpu = bench(pyscamp.selfjoin_c22, ts, window_fixed, threads=HW_THREADS, gpu=True)
        b1_gpu.append(t_gpu)
        row_gpu = {'benchmark': 'scaling_by_length', 'N': n, 'window': window_fixed,
                   'threads': HW_THREADS, 'device': 'GPU', 'time_s': t_gpu}
        all_results.append(row_gpu)
        print(f"  N={n:>6}, GPU: {t_gpu:.3f}s")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, b1_cpu, 'o-', label='CPU', linewidth=2, markersize=6)
    if HAS_GPU and b1_gpu:
        ax.plot(lengths, b1_gpu, 's-', label='GPU', linewidth=2, markersize=6)
    ax.set_xlabel('Time Series Length (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('C22 Profile: Scalability with Time Series Length', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, 'c22_scaling_by_length.png'), dpi=150)
    plt.close(fig)
    print(f"  → Saved c22_scaling_by_length.png")

# ============================================================================
# Benchmark 2: Scalability with window size
# ============================================================================
print()
print("=" * 70)
print("BENCHMARK 2: C22 Profile time vs window size")
print("=" * 70)

n_fixed = 4000
windows = [20, 50, 100, 200, 500]
b2_cpu = []
b2_gpu = []

ts_b2 = make_ts(n_fixed)
for w in windows:
    t_cpu = bench(pyscamp.selfjoin_c22, ts_b2, w, threads=HW_THREADS, gpu=False)
    b2_cpu.append(t_cpu)
    row = {'benchmark': 'scaling_by_window', 'N': n_fixed, 'window': w,
           'threads': HW_THREADS, 'device': 'CPU', 'time_s': t_cpu}
    all_results.append(row)
    print(f"  w={w:>4}, CPU: {t_cpu:.3f}s")

    if HAS_GPU:
        t_gpu = bench(pyscamp.selfjoin_c22, ts_b2, w, threads=HW_THREADS, gpu=True)
        b2_gpu.append(t_gpu)
        row_gpu = {'benchmark': 'scaling_by_window', 'N': n_fixed, 'window': w,
                   'threads': HW_THREADS, 'device': 'GPU', 'time_s': t_gpu}
        all_results.append(row_gpu)
        print(f"  w={w:>4}, GPU: {t_gpu:.3f}s")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(windows, b2_cpu, 'o-', label='CPU', linewidth=2, markersize=6)
    if HAS_GPU and b2_gpu:
        ax.plot(windows, b2_gpu, 's-', label='GPU', linewidth=2, markersize=6)
    ax.set_xlabel('Window Size (m)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'C22 Profile: Scalability with Window Size (N={n_fixed})', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, 'c22_scaling_by_window.png'), dpi=150)
    plt.close(fig)
    print(f"  → Saved c22_scaling_by_window.png")

# ============================================================================
# Benchmark 3: Thread scaling
# ============================================================================
print()
print("=" * 70)
print(f"BENCHMARK 3: C22 Profile thread scaling (max {HW_THREADS} threads)")
print("=" * 70)

n_thread_test = 4000
w_thread_test = 100
thread_counts = [1]
t = 2
while t <= HW_THREADS:
    thread_counts.append(t)
    t *= 2
if HW_THREADS not in thread_counts:
    thread_counts.append(HW_THREADS)

b3_times = []
ts_b3 = make_ts(n_thread_test)

for nthreads in thread_counts:
    t_cpu = bench(pyscamp.selfjoin_c22, ts_b3, w_thread_test, threads=nthreads, gpu=False)
    b3_times.append(t_cpu)
    row = {'benchmark': 'thread_scaling', 'N': n_thread_test, 'window': w_thread_test,
           'threads': nthreads, 'device': 'CPU', 'time_s': t_cpu}
    all_results.append(row)
    print(f"  threads={nthreads:>3}: {t_cpu:.3f}s")

if HAS_MPL:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute time
    ax1.plot(thread_counts, b3_times, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax1.set_xlabel('Number of CPU Threads', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('C22 Profile: Absolute Time vs Threads', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thread_counts)

    # Speedup
    base_time = b3_times[0]  # single-thread
    speedups = [base_time / t for t in b3_times]
    ax2.plot(thread_counts, speedups, 'o-', linewidth=2, markersize=6, color='darkorange',
             label='Actual speedup')
    ax2.plot(thread_counts, thread_counts, '--', color='gray', alpha=0.5, label='Ideal (linear)')
    ax2.set_xlabel('Number of CPU Threads', fontsize=12)
    ax2.set_ylabel('Speedup (vs 1 thread)', fontsize=12)
    ax2.set_title('C22 Profile: Thread Scaling Efficiency', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thread_counts)

    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, 'c22_thread_scaling.png'), dpi=150)
    plt.close(fig)
    print(f"  → Saved c22_thread_scaling.png")

# ============================================================================
# Benchmark 4: Feature computation vs dot-product search breakdown
# ============================================================================
print()
print("=" * 70)
print("BENCHMARK 4: Time breakdown — Feature extraction vs Dot-product search")
print("=" * 70)

# We measure: total time = feature time + search time
# Feature time: use compute_c22_vectors_parallel via a self-join with N=1
# approximation: run self-join on small vs large N to infer the O(N^2) portion

breakdown_N = [1000, 2000, 4000, 8000]
w_break = 100
b4_total = []
b4_feat_est = []
b4_search_est = []

# Estimate feature extraction time by measuring very small N (search is negligible)
# and then extrapolating: feature_time ∝ N, search_time ∝ N^2
ts_small = make_ts(200 + w_break)
t_small = bench(pyscamp.selfjoin_c22, ts_small, w_break, threads=HW_THREADS, gpu=False)
n_small = 200 + w_break - w_break + 1  # 201 subsequences
per_subseq_feat = t_small / n_small  # rough estimate (search is negligible for 201 subseqs)

for n in breakdown_N:
    ts = make_ts(n)
    t_total = bench(pyscamp.selfjoin_c22, ts, w_break, threads=HW_THREADS, gpu=False)
    n_subs = n - w_break + 1
    feat_est = per_subseq_feat * n_subs
    search_est = max(0, t_total - feat_est)

    b4_total.append(t_total)
    b4_feat_est.append(feat_est)
    b4_search_est.append(search_est)

    row = {'benchmark': 'time_breakdown', 'N': n, 'window': w_break,
           'threads': HW_THREADS, 'device': 'CPU', 'time_s': t_total,
           'feat_est_s': feat_est, 'search_est_s': search_est}
    all_results.append(row)
    print(f"  N={n:>6}: total={t_total:.3f}s  (feature≈{feat_est:.3f}s, search≈{search_est:.3f}s)")

if HAS_MPL:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(breakdown_N))
    width = 0.35
    ax.bar(x - width/2, b4_feat_est, width, label='Feature extraction (O(N))',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, b4_search_est, width, label='Dot-product search (O(N²))',
           color='darkorange', alpha=0.8)
    ax.set_xlabel('Time Series Length (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('C22 Profile: Time Breakdown', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in breakdown_N])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, 'c22_time_breakdown.png'), dpi=150)
    plt.close(fig)
    print(f"  → Saved c22_time_breakdown.png")

# ============================================================================
# Combined summary figure
# ============================================================================
if HAS_MPL:
    print()
    print("=" * 70)
    print("Generating combined summary chart ...")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('C22 Profile — Performance Benchmarks', fontsize=15, fontweight='bold')

    # (0,0) Scaling by length
    ax = axes[0, 0]
    ax.plot(lengths, b1_cpu, 'o-', label='CPU', linewidth=2, markersize=5)
    if HAS_GPU and b1_gpu:
        ax.plot(lengths, b1_gpu, 's-', label='GPU', linewidth=2, markersize=5)
    ax.set_xlabel('Time Series Length (N)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Scalability with N')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # (0,1) Scaling by window
    ax = axes[0, 1]
    ax.plot(windows, b2_cpu, 'o-', label='CPU', linewidth=2, markersize=5)
    if HAS_GPU and b2_gpu:
        ax.plot(windows, b2_gpu, 's-', label='GPU', linewidth=2, markersize=5)
    ax.set_xlabel('Window Size (m)')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Scalability with Window (N={n_fixed})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,0) Thread scaling
    ax = axes[1, 0]
    base_time = b3_times[0]
    speedups = [base_time / t for t in b3_times]
    ax.plot(thread_counts, speedups, 'o-', linewidth=2, markersize=5, color='darkorange',
            label='Actual')
    ax.plot(thread_counts, thread_counts, '--', color='gray', alpha=0.5, label='Ideal')
    ax.set_xlabel('Threads')
    ax.set_ylabel('Speedup')
    ax.set_title('Thread Scaling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(thread_counts)

    # (1,1) Time breakdown
    ax = axes[1, 1]
    x = np.arange(len(breakdown_N))
    width = 0.35
    ax.bar(x - width/2, b4_feat_est, width, label='Features (O(N))',
           color='steelblue', alpha=0.8)
    ax.bar(x + width/2, b4_search_est, width, label='Search (O(N²))',
           color='darkorange', alpha=0.8)
    ax.set_xlabel('Time Series Length (N)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Time Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in breakdown_N])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(CHART_DIR, 'c22_benchmark_summary.png'), dpi=150)
    plt.close(fig)
    print(f"  → Saved c22_benchmark_summary.png")

# ============================================================================
# Save CSV
# ============================================================================
csv_path = os.path.join(CHART_DIR, 'benchmark_results.csv')
if all_results:
    # Collect all keys
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  → Raw data saved to {csv_path}")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 70)
print("BENCHMARK COMPLETE")
print(f"  Hardware threads: {HW_THREADS}")
print(f"  GPU available:    {HAS_GPU}")
print(f"  Charts saved to:  {CHART_DIR}/")
if HAS_MPL:
    print("  Files: c22_scaling_by_length.png, c22_scaling_by_window.png,")
    print("         c22_thread_scaling.png, c22_time_breakdown.png,")
    print("         c22_benchmark_summary.png, benchmark_results.csv")
else:
    print("  Files: benchmark_results.csv (install matplotlib for charts)")
print("=" * 70)
