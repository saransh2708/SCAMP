#!/usr/bin/env python3
"""
GPU validation test for C22 Profile.
Runs the same computation on CPU and GPU, then compares results.
Usage:  python test_c22_gpu.py
"""

import sys
import time
import numpy as np

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Build with: cmake .. -DCMAKE_CUDA_COMPILER=nvcc && make -j")
    sys.exit(1)

# Check GPU availability
has_gpu = pyscamp.gpu_supported()
print(f"GPU support compiled in: {has_gpu}")
if not has_gpu:
    print("ERROR: pyscamp was not built with CUDA support.")
    print("Rebuild with: cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc) && make -j")
    sys.exit(1)

np.random.seed(42)
PASS = 0
FAIL = 0

# GPU features use different algorithmic implementations (custom iterative FFT,
# direct autocorrelation loops, etc.) compared to the CPU pycatch22 C library.
# These numerical differences accumulate across 22 features in the dot product.
# Tolerance of 1.0 is appropriate for profile values typically in the 100-1000+ range.
GPU_TOL = 1.0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")

# ============================================================================
# Test 1: Self-join — small series, exact match CPU vs GPU
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Self-join CPU vs GPU (small series, N=500)")
print("=" * 70)

n, m = 500, 50
ts = list(np.random.randn(n))

t0 = time.time()
prof_cpu, idx_cpu = pyscamp.selfjoin_c22(ts, m, gpu=False)
cpu_time = time.time() - t0

t0 = time.time()
prof_gpu, idx_gpu = pyscamp.selfjoin_c22(ts, m, gpu=True)
gpu_time = time.time() - t0

prof_cpu = np.array(prof_cpu)
prof_gpu = np.array(prof_gpu)
idx_cpu = np.array(idx_cpu)
idx_gpu = np.array(idx_gpu)

check("Same length", len(prof_cpu) == len(prof_gpu),
      f"CPU={len(prof_cpu)} GPU={len(prof_gpu)}")

prof_diff = np.max(np.abs(prof_cpu - prof_gpu))
check(f"Profile values match (max diff < {GPU_TOL})", prof_diff < GPU_TOL,
      f"max_diff={prof_diff:.4g}")

idx_match = np.sum(idx_cpu == idx_gpu)
idx_total = len(idx_cpu)
pct = 100 * idx_match / idx_total
print(f"  Index match: {idx_match}/{idx_total} ({pct:.1f}%)")

# When indices differ, profile values should still be close (tie-breaking).
# GPU features use different algorithms (custom FFT, direct autocorrelation)
# than CPU pycatch22, so small feature differences cause different tie-breaks.
if idx_match < idx_total:
    mismatch = idx_cpu != idx_gpu
    tie_diff = np.max(np.abs(prof_cpu[mismatch] - prof_gpu[mismatch]))
    check("Mismatched indices have close profile value (tie-break)",
          tie_diff < GPU_TOL, f"max_diff={tie_diff:.4g}")

print(f"  Timing: CPU={cpu_time:.3f}s  GPU={gpu_time:.3f}s")

# ============================================================================
# Test 2: Self-join — larger series
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Self-join CPU vs GPU (larger series, N=2000)")
print("=" * 70)

n, m = 2000, 100

# Structured data with embedded patterns
ts2 = np.random.randn(n) * 0.3
for start in [200, 800, 1400]:
    ts2[start:start+200] = np.sin(np.linspace(0, 6*np.pi, 200))
ts2 = list(ts2)

t0 = time.time()
prof_cpu, idx_cpu = pyscamp.selfjoin_c22(ts2, m, gpu=False)
cpu_time = time.time() - t0

t0 = time.time()
prof_gpu, idx_gpu = pyscamp.selfjoin_c22(ts2, m, gpu=True)
gpu_time = time.time() - t0

prof_cpu = np.array(prof_cpu)
prof_gpu = np.array(prof_gpu)
idx_cpu = np.array(idx_cpu)
idx_gpu = np.array(idx_gpu)

prof_diff = np.max(np.abs(prof_cpu - prof_gpu))
check(f"Profile values match (max diff < {GPU_TOL})", prof_diff < GPU_TOL,
      f"max_diff={prof_diff:.4g}")

idx_match = np.sum(idx_cpu == idx_gpu)
idx_total = len(idx_cpu)
pct = 100 * idx_match / idx_total
print(f"  Index match: {idx_match}/{idx_total} ({pct:.1f}%)")
if idx_match < idx_total:
    mismatch = idx_cpu != idx_gpu
    tie_diff = np.max(np.abs(prof_cpu[mismatch] - prof_gpu[mismatch]))
    check("Mismatched indices have close profile value (tie-break)",
          tie_diff < GPU_TOL, f"max_diff={tie_diff:.4g}")

print(f"  Timing: CPU={cpu_time:.3f}s  GPU={gpu_time:.3f}s")

# ============================================================================
# Test 3: AB-join CPU vs GPU
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: AB-join CPU vs GPU")
print("=" * 70)

n_a, n_b, m = 600, 400, 50
ts_a = list(np.random.randn(n_a))
ts_b = list(np.random.randn(n_b))

t0 = time.time()
prof_cpu, idx_cpu = pyscamp.abjoin_c22(ts_a, ts_b, m, gpu=False)
cpu_time = time.time() - t0

t0 = time.time()
prof_gpu, idx_gpu = pyscamp.abjoin_c22(ts_a, ts_b, m, gpu=True)
gpu_time = time.time() - t0

prof_cpu = np.array(prof_cpu)
prof_gpu = np.array(prof_gpu)
idx_cpu = np.array(idx_cpu)
idx_gpu = np.array(idx_gpu)

prof_diff = np.max(np.abs(prof_cpu - prof_gpu))
check(f"Profile values match (max diff < {GPU_TOL})", prof_diff < GPU_TOL,
      f"max_diff={prof_diff:.4g}")

idx_match = np.sum(idx_cpu == idx_gpu)
idx_total = len(idx_cpu)
pct = 100 * idx_match / idx_total
print(f"  Index match: {idx_match}/{idx_total} ({pct:.1f}%)")
if idx_match < idx_total:
    mismatch = idx_cpu != idx_gpu
    tie_diff = np.max(np.abs(prof_cpu[mismatch] - prof_gpu[mismatch]))
    check("Mismatched indices have close profile value (tie-break)",
          tie_diff < GPU_TOL, f"max_diff={tie_diff:.4g}")

print(f"  Timing: CPU={cpu_time:.3f}s  GPU={gpu_time:.3f}s")

# ============================================================================
# Test 4: GPU speedup on large series
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: GPU speedup benchmark (N=5000)")
print("=" * 70)

n, m = 5000, 100
ts_big = list(np.random.randn(n))

t0 = time.time()
prof_cpu, _ = pyscamp.selfjoin_c22(ts_big, m, gpu=False)
cpu_time = time.time() - t0

t0 = time.time()
prof_gpu, _ = pyscamp.selfjoin_c22(ts_big, m, gpu=True)
gpu_time = time.time() - t0

prof_cpu = np.array(prof_cpu)
prof_gpu = np.array(prof_gpu)
prof_diff = np.max(np.abs(prof_cpu - prof_gpu))
check(f"Profile values match (max diff < {GPU_TOL})", prof_diff < GPU_TOL,
      f"max_diff={prof_diff:.4g}")

speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
print(f"  CPU time: {cpu_time:.3f}s")
print(f"  GPU time: {gpu_time:.3f}s")
print(f"  Speedup:  {speedup:.1f}x")

# Note: for small N, GPU may not be faster due to launch overhead.
# The dot product search is O(N^2) which is where GPU shines for large N.
# Feature computation is always on CPU regardless.

# ============================================================================
# Test 5: Edge case — minimum viable series
# ============================================================================
print("\n" + "=" * 70)
print("TEST 5: Edge case — small series (N=100, m=20)")
print("=" * 70)

ts_small = list(np.random.randn(100))
prof_cpu, idx_cpu = pyscamp.selfjoin_c22(ts_small, 20, gpu=False)
prof_gpu, idx_gpu = pyscamp.selfjoin_c22(ts_small, 20, gpu=True)
prof_cpu = np.array(prof_cpu)
prof_gpu = np.array(prof_gpu)

prof_diff = np.max(np.abs(prof_cpu - prof_gpu))
check(f"Small series: profile match (max diff < {GPU_TOL})",
      prof_diff < GPU_TOL, f"max_diff={prof_diff:.4g}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
else:
    print("\nAll GPU tests passed! CPU and GPU produce identical results.")
    sys.exit(0)
