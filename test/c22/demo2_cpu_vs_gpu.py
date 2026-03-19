#!/usr/bin/env python3
"""
Demo 2: C22 Profile — CPU vs GPU Performance
Compares best CPU (all threads) against GPU on the same dataset.

Run on cerebro:
  PYTHONPATH=build/src/python python3 test/demo2_cpu_vs_gpu.py
"""
import csv
import time
import multiprocessing
import pyscamp

# ── Load ECG dataset (~45K points) ──
ts = []
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', '..', '20Papers ', 'ecg.csv')
with open(DATA_PATH) as f:
    for row in csv.DictReader(f):
        ts.append(float(row['ts']))

N = len(ts)
W = 100
N_SUB = N - W + 1
MAX_THREADS = multiprocessing.cpu_count()

print("=" * 60)
print("  C²² Profile — CPU vs GPU Performance Demo")
print("=" * 60)
print(f"  Dataset:        ECG (N = {N:,})")
print(f"  Window:         {W}")
print(f"  Subsequences:   {N_SUB:,}")
print(f"  CPU threads:    {MAX_THREADS}")
print(f"  GPU available:  {pyscamp.gpu_supported()}")
print("=" * 60)

# ── CPU benchmark (all threads) ──
print(f"\n  Running CPU ({MAX_THREADS} threads)...", end=" ", flush=True)
t0 = time.time()
c22_cpu, idx_cpu = pyscamp.selfjoin_c22(ts, W, threads=MAX_THREADS, gpu=False)
cpu_time = time.time() - t0
print(f"{cpu_time:.2f}s")

# ── GPU benchmark ──
print(f"  Running GPU...", end=" ", flush=True)
t0 = time.time()
c22_gpu, idx_gpu = pyscamp.selfjoin_c22(ts, W, threads=MAX_THREADS, gpu=True)
gpu_time = time.time() - t0
print(f"{gpu_time:.2f}s")

# ── Results ──
speedup = cpu_time / gpu_time
pairs = N_SUB * (N_SUB - 1) / 2
cpu_throughput = pairs / cpu_time / 1e6
gpu_throughput = pairs / gpu_time / 1e6

print()
print("  ┌─────────────────────────────────────────┐")
print(f"  │  CPU time:        {cpu_time:>8.2f}s              │")
print(f"  │  GPU time:        {gpu_time:>8.2f}s              │")
print(f"  │  GPU speedup:     {speedup:>8.2f}x              │")
print(f"  │                                         │")
print(f"  │  CPU throughput:  {cpu_throughput:>8.1f} M pairs/s    │")
print(f"  │  GPU throughput:  {gpu_throughput:>8.1f} M pairs/s    │")
print("  └─────────────────────────────────────────┘")

# ── Verify correctness: profiles should match ──
mismatches = sum(1 for a, b in zip(idx_cpu, idx_gpu) if a != b)
match_pct = (1 - mismatches / len(idx_cpu)) * 100
print(f"\n  Index agreement: {match_pct:.1f}% ({len(idx_cpu) - mismatches}/{len(idx_cpu)})")

print()
print("Done!")
