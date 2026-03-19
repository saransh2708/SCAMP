#!/usr/bin/env python3
"""
Demo 1: C22 Profile — CPU Thread Scaling
Shows how C22 Profile computation scales with number of CPU threads.

Run on cerebro:
  PYTHONPATH=build/src/python python3 test/demo1_thread_scaling.py
"""
import csv
import time
import multiprocessing
import pyscamp

# ── Load ECG dataset (~45K points) ──
ts = []
with open('20Papers /ecg.csv') as f:
    for row in csv.DictReader(f):
        ts.append(float(row['ts']))

N = len(ts)
W = 100
MAX_THREADS = multiprocessing.cpu_count()

# Thread counts: 1, 2, 4, 8, ..., up to max
thread_counts = [1]
t = 2
while t <= MAX_THREADS:
    thread_counts.append(t)
    t *= 2
if MAX_THREADS not in thread_counts:
    thread_counts.append(MAX_THREADS)

print("=" * 60)
print("  C²² Profile — Thread Scaling Demo")
print("=" * 60)
print(f"  Dataset:     ECG (N = {N:,})")
print(f"  Window:      {W}")
print(f"  Max threads: {MAX_THREADS}")
print("=" * 60)
print()
print(f"  {'Threads':>8}  {'Time (s)':>10}  {'Speedup':>10}  {'Efficiency':>12}")
print("  " + "-" * 44)

base_time = None
for tc in thread_counts:
    t0 = time.time()
    _ = pyscamp.selfjoin_c22(ts, W, threads=tc, gpu=False)
    elapsed = time.time() - t0

    if base_time is None:
        base_time = elapsed

    speedup = base_time / elapsed
    efficiency = speedup / tc * 100

    print(f"  {tc:>8}  {elapsed:>9.2f}s  {speedup:>9.2f}x  {efficiency:>10.1f}%")

print()
print("  Speedup = T(1 thread) / T(N threads)")
print("  Efficiency = Speedup / N_threads × 100%")
print("  (100% = perfect linear scaling)")
print()
print("Done!")
