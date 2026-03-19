#!/usr/bin/env python3
"""
Test C22 Profile on pycatch22's benchmark time series.

Uses the real time series from pycatch22/tests/benchmarks/inputs/ to verify
that the C22 Profile produces meaningful results on realistic data.
"""

import sys
import os
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'src', 'python'))
import pyscamp

BENCHMARK_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'pycatch22', 'tests', 'benchmarks', 'inputs'
)

def load_ts(filename):
    """Load a time series from the benchmark inputs directory."""
    path = os.path.join(BENCHMARK_DIR, filename)
    with open(path, 'r') as f:
        content = f.read().strip()
    # Handle both one-per-line and space-separated formats
    values = content.replace('\n', ' ').split()
    return [float(v) for v in values]

# ============================================================
# Load all benchmark time series
# ============================================================
print("=" * 60)
print("C22 Profile on pycatch22 Benchmark Time Series")
print("=" * 60)
print()

ts_test     = load_ts('test_input.txt')        # 270 points
ts_test2    = load_ts('test2_input.txt')        # 270 points (same data, space-separated)
ts_sinusoid = load_ts('testSinusoid_input.txt') # 5001 points
ts_short    = load_ts('testShort_input.txt')    # 12 points

# NaN/Inf series — replace invalid values for C22 (catch22 handles internally)
ts_nan      = load_ts('testNaN_input.txt')      # 260 points, first=NaN
ts_inf      = load_ts('testInf_input.txt')      # 260 points, first=Inf
ts_inf_neg  = load_ts('testInfMinus_input.txt') # 260 points, first=-Inf

print(f"Loaded time series:")
print(f"  test_input:       {len(ts_test)} points")
print(f"  test2_input:      {len(ts_test2)} points")
print(f"  testSinusoid:     {len(ts_sinusoid)} points")
print(f"  testShort:        {len(ts_short)} points")
print(f"  testNaN:          {len(ts_nan)} points")
print(f"  testInf:          {len(ts_inf)} points")
print(f"  testInfMinus:     {len(ts_inf_neg)} points")
print()

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}" + (f" — {detail}" if detail else ""))
        passed += 1
    else:
        print(f"  ✗ {name}" + (f" — {detail}" if detail else ""))
        failed += 1

# ============================================================
# Test 1: Self-join on test_input (270 points, window=50)
# ============================================================
print("Test 1: Self-join on test_input (270 pts, window=50)")
m = 50
prof, idx = pyscamp.selfjoin_c22(ts_test, m)
n = len(ts_test) - m + 1
check("Profile length correct", len(prof) == n, f"expected {n}, got {len(prof)}")
check("Index length correct", len(idx) == n, f"expected {n}, got {len(idx)}")
check("All indices valid", all(0 <= i < n for i in idx),
      f"range: [{min(idx)}, {max(idx)}]")
check("All profile values finite", all(np.isfinite(p) for p in prof),
      f"range: [{min(prof):.4f}, {max(prof):.4f}]")
# Exclusion zone: no self-match within window/4
ez = m // 4
violations = [i for i in range(n) if abs(i - idx[i]) <= ez]
check("Exclusion zone respected", len(violations) == 0,
      f"{len(violations)} violations" if violations else "no violations")
print()

# ============================================================
# Test 2: Self-join on sinusoid (5001 points, window=100)
# ============================================================
print("Test 2: Self-join on sinusoid (5001 pts, window=100)")
m = 100
prof_sin, idx_sin = pyscamp.selfjoin_c22(ts_sinusoid, m)
n_sin = len(ts_sinusoid) - m + 1
check("Profile length correct", len(prof_sin) == n_sin,
      f"expected {n_sin}, got {len(prof_sin)}")
check("All indices valid", all(0 <= i < n_sin for i in idx_sin))
check("All profile values finite", all(np.isfinite(p) for p in prof_sin))

# For a sinusoid, profile values should be relatively high and consistent
prof_arr = np.array(prof_sin)
mean_prof = np.mean(prof_arr)
std_prof = np.std(prof_arr)
cv = std_prof / abs(mean_prof) if abs(mean_prof) > 1e-12 else float('inf')
check("Profile values consistent for periodic signal", cv < 1.0,
      f"CV={cv:.4f}, mean={mean_prof:.4f}, std={std_prof:.4f}")
print()

# ============================================================
# Test 3: test_input and test2_input should be identical data
# ============================================================
print("Test 3: test_input vs test2_input (same data, different format)")
check("Data is identical", np.allclose(ts_test, ts_test2),
      f"max diff={max(abs(a-b) for a,b in zip(ts_test,ts_test2)):.2e}")

prof2, idx2 = pyscamp.selfjoin_c22(ts_test2, 50)
check("Profiles match", np.allclose(prof, prof2),
      f"max diff={max(abs(a-b) for a,b in zip(prof,prof2)):.2e}")
check("Indices match", idx == idx2)
print()

# ============================================================
# Test 4: AB-join between test_input and sinusoid
# ============================================================
print("Test 4: AB-join between test_input and sinusoid")
m = 50
prof_ab, idx_ab = pyscamp.abjoin_c22(ts_test, ts_sinusoid, m)
na = len(ts_test) - m + 1
nb = len(ts_sinusoid) - m + 1
check("Profile length = num subsequences in A", len(prof_ab) == na,
      f"expected {na}, got {len(prof_ab)}")
check("Index length correct", len(idx_ab) == na)
check("All indices point into B", all(0 <= i < nb for i in idx_ab),
      f"range: [{min(idx_ab)}, {max(idx_ab)}], B has {nb} subseqs")
check("All profile values finite", all(np.isfinite(p) for p in prof_ab))
print()

# ============================================================
# Test 5: AB-join test_input against itself = same as self-join (no excl zone)
# ============================================================
print("Test 5: AB-join A against A (dot product properties)")
m = 50
prof_aa, idx_aa = pyscamp.abjoin_c22(ts_test, ts_test, m)
na = len(ts_test) - m + 1
# Note: with raw dot products (not cosine similarity), the self-match
# F_i·F_i is NOT guaranteed to be the max — F_i·F_j can exceed F_i·F_i
# when ||F_j|| > ||F_i||. Instead, verify:
# 1) Profile values ≥ 0 (dot product with similar features should be positive)
check("All AB-join profile values non-negative",
      all(p >= 0 for p in prof_aa),
      f"range: [{min(prof_aa):.3f}, {max(prof_aa):.3f}]")
# 2) AB-join profile ≥ self-join profile (AB has no exclusion zone → more candidates)
check("AB(A,A) ≥ selfjoin(A) for all i (no exclusion → at least as good)",
      all(pab >= psj - 1e-9 for pab, psj in zip(prof_aa, prof)),
      f"min margin: {min(pab - psj for pab, psj in zip(prof_aa, prof)):.6f}")
# 3) All indices valid
check("All indices valid", all(0 <= i < na for i in idx_aa))
print()

# ============================================================
# Test 6: Handle NaN input gracefully
# ============================================================
print("Test 6: Self-join on NaN-containing series (260 pts, window=30)")
m = 30
try:
    prof_nan, idx_nan = pyscamp.selfjoin_c22(ts_nan, m)
    n_nan = len(ts_nan) - m + 1
    check("Completed without crash", True)
    check("Profile length correct", len(prof_nan) == n_nan)
    check("All indices valid", all(0 <= i < n_nan for i in idx_nan))
    # Profile values may contain unusual values due to NaN in input
    finite_count = sum(1 for p in prof_nan if np.isfinite(p))
    check("Most profile values finite", finite_count / n_nan > 0.8,
          f"{finite_count}/{n_nan} finite")
except Exception as e:
    check("Completed without crash", False, str(e))
print()

# ============================================================
# Test 7: Handle Inf input gracefully
# ============================================================
print("Test 7: Self-join on Inf-containing series (260 pts, window=30)")
m = 30
try:
    prof_inf, idx_inf = pyscamp.selfjoin_c22(ts_inf, m)
    n_inf = len(ts_inf) - m + 1
    check("Completed without crash", True)
    check("Profile length correct", len(prof_inf) == n_inf)
    check("All indices valid", all(0 <= i < n_inf for i in idx_inf))
    finite_count = sum(1 for p in prof_inf if np.isfinite(p))
    check("Most profile values finite", finite_count / n_inf > 0.8,
          f"{finite_count}/{n_inf} finite")
except Exception as e:
    check("Completed without crash", False, str(e))
print()

# ============================================================
# Test 8: Handle -Inf input gracefully
# ============================================================
print("Test 8: Self-join on -Inf-containing series (260 pts, window=30)")
m = 30
try:
    prof_infn, idx_infn = pyscamp.selfjoin_c22(ts_inf_neg, m)
    n_infn = len(ts_inf_neg) - m + 1
    check("Completed without crash", True)
    check("Profile length correct", len(prof_infn) == n_infn)
    check("All indices valid", all(0 <= i < n_infn for i in idx_infn))
    finite_count = sum(1 for p in prof_infn if np.isfinite(p))
    check("Most profile values finite", finite_count / n_infn > 0.8,
          f"{finite_count}/{n_infn} finite")
except Exception as e:
    check("Completed without crash", False, str(e))
print()

# ============================================================
# Test 9: Short series — should handle gracefully
# ============================================================
print("Test 9: Self-join on short series (12 pts, window=5)")
m = 5
try:
    prof_s, idx_s = pyscamp.selfjoin_c22(ts_short, m)
    n_s = len(ts_short) - m + 1
    check("Profile length correct", len(prof_s) == n_s,
          f"expected {n_s}, got {len(prof_s)}")
    check("All indices valid", all(0 <= i < n_s for i in idx_s))
    ez_s = m // 4
    violations_s = [i for i in range(n_s) if abs(i - idx_s[i]) <= ez_s]
    check("Exclusion zone respected", len(violations_s) == 0)
except Exception as e:
    check("Completed without crash", False, str(e))
print()

# ============================================================
# Test 10: AB-join between sinusoid and NaN series
# ============================================================
print("Test 10: AB-join sinusoid→NaN series (cross-series robustness)")
m = 30
try:
    prof_cross, idx_cross = pyscamp.abjoin_c22(ts_sinusoid[:500], ts_nan, m)
    na = len(ts_sinusoid[:500]) - m + 1
    nb = len(ts_nan) - m + 1
    check("Completed without crash", True)
    check("Profile length correct", len(prof_cross) == na)
    check("All indices point into B", all(0 <= i < nb for i in idx_cross))
except Exception as e:
    check("Completed without crash", False, str(e))
print()

# ============================================================
# Test 11: Sinusoid self-join — verify nearest neighbor offsets
# ============================================================
print("Test 11: Sinusoid nearest-neighbor offset analysis")
m = 100
# For a periodic sinusoid, the nearest neighbor should be offset by the period
offsets = np.abs(np.array(idx_sin) - np.arange(n_sin))
median_offset = int(np.median(offsets))
# The sinusoid test file should have some periodicity — check if offsets cluster
offset_std = np.std(offsets)
check("Offsets have structure (not random)", offset_std < n_sin / 3,
      f"offset std={offset_std:.1f}, median={median_offset}")
# Most offsets should be beyond exclusion zone
ez_sin = m // 4
beyond_ez = sum(1 for o in offsets if o > ez_sin)
check("All offsets beyond exclusion zone", beyond_ez == n_sin,
      f"{beyond_ez}/{n_sin}")
print()

# ============================================================
# Test 12: Window size sensitivity on test_input
# ============================================================
print("Test 12: Window size sensitivity on test_input")
windows = [20, 50, 100]
profiles_by_window = {}
for w in windows:
    if w < len(ts_test):
        p, idx_w = pyscamp.selfjoin_c22(ts_test, w)
        profiles_by_window[w] = (p, idx_w)
        n_w = len(ts_test) - w + 1
        check(f"window={w}: length={n_w}, range=[{min(p):.3f}, {max(p):.3f}]",
              len(p) == n_w and all(np.isfinite(v) for v in p))

# Different window sizes should produce different profiles
if 20 in profiles_by_window and 100 in profiles_by_window:
    p20 = profiles_by_window[20][0]
    p100 = profiles_by_window[100][0]
    # They have different lengths, so just check they exist and are valid
    check("Different windows → different profile lengths",
          len(p20) != len(p100),
          f"w=20: {len(p20)} pts, w=100: {len(p100)} pts")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
total = passed + failed
if failed == 0:
    print(f"ALL {passed} CHECKS PASSED")
else:
    print(f"{passed}/{total} checks passed, {failed} FAILED")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
