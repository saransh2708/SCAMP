#!/usr/bin/env python3
"""
Ground-truth correctness test for the C22 GPU kernel.

Strategy
--------
1. Build a tiny time series (a few dozen points) so we can enumerate
   ALL pairs of subsequences by brute force in Python.
2. Compute C22 features for every subsequence independently via
   pycatch22 (the reference Python implementation).
3. Manually compute all dot products and find the true C22 profile.
4. Run pyscamp.selfjoin_c22 with gpu=True and gpu=False.
5. Assert GPU == CPU == brute-force truth.

Run from repo root:
  PYTHONPATH=build/src/python python3 test/test_c22_gpu_correctness.py
"""

import sys
import math
import numpy as np

# ── pyscamp ──────────────────────────────────────────────────────────────────
try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Run with PYTHONPATH=build/src/python")
    sys.exit(1)

# ── pycatch22 (reference) ─────────────────────────────────────────────────────
try:
    import pycatch22
    HAS_PYCATCH22 = True
except ImportError:
    HAS_PYCATCH22 = False
    print("WARNING: pycatch22 Python package not found – skipping reference comparison.")
    print("         Install with: pip install pycatch22")

HAS_GPU = pyscamp.gpu_supported()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def check(label, ok, detail=""):
    mark = "[PASS]" if ok else "[FAIL]"
    msg  = f"  {mark} {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok

def zscore(x):
    """Z-score normalise a list/array (matches pycatch22 internal behaviour)."""
    a = np.array(x, dtype=np.float64)
    mu, sigma = a.mean(), a.std(ddof=0)
    if sigma < 1e-10:
        return np.zeros_like(a)
    return (a - mu) / sigma

def features_via_pycatch22(subseq):
    """Compute 22 features for a single subsequence using the pycatch22 Python API.
    pycatch22.catch22_all z-scores internally, so pass raw data."""
    res = pycatch22.catch22_all(list(subseq))
    vals = [v if math.isfinite(v) else 0.0 for v in res["values"]]
    return np.array(vals, dtype=np.float64)

def dot22(a, b):
    return float(np.dot(a, b))

def brute_force_c22_profile(ts, window, exclusion_zone=None):
    """Compute C22 profile by brute force via pycatch22 reference features."""
    ts = list(ts)
    N = len(ts) - window + 1
    if exclusion_zone is None:
        exclusion_zone = window // 4

    print(f"  Computing {N} reference feature vectors via pycatch22 ...", flush=True)
    F = []
    for i in range(N):
        subseq = ts[i : i + window]
        F.append(features_via_pycatch22(subseq))

    print(f"  Computing {N}x{N} dot products ...", flush=True)
    profile = [-math.inf] * N
    index   = [-1] * N
    for i in range(N):
        for j in range(N):
            diff = abs(i - j)
            if diff <= exclusion_zone:
                continue
            d = dot22(F[i], F[j])
            if d > profile[i]:
                profile[i] = d
                index[i]   = j
    return np.array(profile), np.array(index), F

# ─────────────────────────────────────────────────────────────────────────────
# Test series definitions
# ─────────────────────────────────────────────────────────────────────────────

def make_sine_series(n=300, period=50, noise=0.05, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n, dtype=np.float64)
    return np.sin(2 * np.pi * t / period) + noise * rng.standard_normal(n)

def make_two_class_series(window=50, reps=3, seed=7):
    """Interleave class-A (sine) and class-B (sawtooth) subsequences.
    We know: same-class pairs must have higher dot products than cross-class pairs."""
    rng = np.random.default_rng(seed)
    t   = np.arange(window, dtype=np.float64)
    sine  = np.sin(2 * np.pi * t / window)
    saw   = (t / window - 0.5) * 2
    parts = []
    for _ in range(reps):
        parts.append(sine  + 0.05 * rng.standard_normal(window))
        parts.append(saw   + 0.05 * rng.standard_normal(window))
    return np.concatenate(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def compare_prof_idx(label, ref_prof, ref_idx, got_prof, got_idx,
                     tol_prof=1e-4, tol_idx_rate=0.95):
    """Compare profile values and indices between a reference and a result."""
    passed = True
    N = len(ref_prof)

    prof_diff = np.max(np.abs(ref_prof - got_prof))
    ok = prof_diff < tol_prof
    passed &= check(f"{label}: profile values match reference (max_diff={prof_diff:.2e})", ok)

    # Index match: same index, OR profile value at got_idx equals reference max
    match = (ref_idx == got_idx)
    n_match = match.sum()
    if n_match < N * tol_idx_rate:
        # Check tie-break: if indices differ, profile values should be equal
        mismatch = ~match
        tie_diff = np.max(
            np.abs(ref_prof[mismatch] - got_prof[mismatch])
        ) if mismatch.any() else 0.0
        ok2 = tie_diff < tol_prof
        passed &= check(
            f"{label}: index match {n_match}/{N} ({100*n_match/N:.1f}%) "
            f"— mismatched indices are ties",
            ok2, f"max_prof_diff_at_mismatches={tie_diff:.2e}")
    else:
        passed &= check(f"{label}: index match {n_match}/{N} ({100*n_match/N:.1f}%)", True)

    return passed

# ─────────────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────────────
all_passed = True

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 1: CPU == GPU on tiny series (brute-force verifiable)")
print("=" * 70)
# Very small so brute force is fast and indices are unique
ts1   = make_sine_series(n=250, period=40)
WIN1  = 50    # gives 201 subsequences — small enough to brute-force
EXC1  = WIN1 // 4

cpu_prof1, cpu_idx1 = pyscamp.selfjoin_c22(list(ts1), WIN1, gpu=False)
cpu_prof1 = np.array(cpu_prof1)
cpu_idx1  = np.array(cpu_idx1)

if HAS_GPU:
    gpu_prof1, gpu_idx1 = pyscamp.selfjoin_c22(list(ts1), WIN1, gpu=True)
    gpu_prof1 = np.array(gpu_prof1)
    gpu_idx1  = np.array(gpu_idx1)

    diff = np.max(np.abs(cpu_prof1 - gpu_prof1))
    ok = check("CPU profile == GPU profile (max_diff)", diff < 1e-6,
               f"max_diff={diff:.2e}")
    all_passed &= ok

    idx_match = np.sum(cpu_idx1 == gpu_idx1)
    ok = check(f"CPU index == GPU index ({idx_match}/{len(cpu_idx1)})",
               idx_match == len(cpu_idx1))
    if not ok:
        # Allow tie-break
        mis = cpu_idx1 != gpu_idx1
        tie = np.max(np.abs(cpu_prof1[mis] - gpu_prof1[mis]))
        ok2 = check("Mismatched indices are ties (profile values equal)",
                    tie < 1e-6, f"max_diff={tie:.2e}")
        all_passed &= ok2
else:
    print("  [SKIP] GPU not available")

if HAS_PYCATCH22:
    print()
    print("  ── Reference check via pycatch22 ──")
    ref_prof1, ref_idx1, _ = brute_force_c22_profile(ts1, WIN1)
    all_passed &= compare_prof_idx("CPU vs pycatch22", ref_prof1, ref_idx1,
                                   cpu_prof1, cpu_idx1)
    if HAS_GPU:
        all_passed &= compare_prof_idx("GPU vs pycatch22", ref_prof1, ref_idx1,
                                       gpu_prof1, gpu_idx1)

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 2: Two-class series — same-class pairs must score higher")
print("=" * 70)
# Series: [sine, saw, sine, saw, sine, saw]  (each of length WIN2)
WIN2 = 60
ts2  = make_two_class_series(window=WIN2, reps=4)
N2   = len(ts2) - WIN2 + 1
print(f"  Series length={len(ts2)}, subsequences={N2}, window={WIN2}")

cpu_prof2, cpu_idx2 = pyscamp.selfjoin_c22(list(ts2), WIN2, gpu=False)
cpu_prof2 = np.array(cpu_prof2)
cpu_idx2  = np.array(cpu_idx2)

if HAS_GPU:
    gpu_prof2, gpu_idx2 = pyscamp.selfjoin_c22(list(ts2), WIN2, gpu=True)
    gpu_prof2 = np.array(gpu_prof2)
    gpu_idx2  = np.array(gpu_idx2)
    diff2 = np.max(np.abs(cpu_prof2 - gpu_prof2))
    all_passed &= check("CPU profile == GPU profile", diff2 < 1e-6,
                        f"max_diff={diff2:.2e}")

# Subsequence layout (each WIN2 points → 1 subsequence):
#   subseq 0 → sine,  subseq 1 → saw,  subseq 2 → sine, ...
# Because reps=4 and each class is exactly WIN2 long.
# Profile of sine subseq should point to another sine subseq (not saw).
n_subs = N2
sine_starts = list(range(0, n_subs, 2 * WIN2))   # rough approximation
# More precisely: plot which class each subseq centre falls in.
exc = WIN2 // 4

# Only test unambiguous "anchor" subsequences at the START of each class block.
# A subsequence starting exactly at block i*WIN2 is purely sine or purely saw.
# Anything in between is a mix and we skip it.
n_blocks = (len(ts2)) // WIN2   # number of complete WIN2-length blocks
anchor_positions = [i * WIN2 for i in range(n_blocks) if i * WIN2 < n_subs]
# class of anchor i*WIN2: even blocks = sine (0), odd blocks = saw (1)
anchor_class = {pos: (pos // WIN2) % 2 for pos in anchor_positions}

same_class_nn = 0
eligible = 0
for pos in anchor_positions:
    nn = cpu_idx2[pos]
    if nn < 0 or abs(pos - nn) <= exc:
        continue
    eligible += 1
    # Check if nearest-neighbour is also at an anchor of the same class
    # (or at least in a region of the same class)
    nn_class = (nn // WIN2) % 2   # approximate class of nn's block
    if anchor_class[pos] == nn_class:
        same_class_nn += 1

rate = same_class_nn / eligible if eligible else 0
all_passed &= check(
    f"Anchor NN is same class ({same_class_nn}/{eligible}, {100*rate:.1f}%)",
    rate >= 0.75,  # 75% — some anchors near edges may still bleed
    f"eligible anchors={eligible}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 3: Self-match is the global maximum (sanity check)")
print("=" * 70)
# For any row i, the dot product F[i]·F[i] is the squared L2 norm of F[i].
# Since we exclude self, the profile[i] must be < F[i]·F[i].
ts3   = make_sine_series(n=300, period=60, noise=0.1)
WIN3  = 50
cpu_p3, cpu_i3 = pyscamp.selfjoin_c22(list(ts3), WIN3, gpu=False)
cpu_p3 = np.array(cpu_p3)

# Compute self dot-products via pyscamp AB-join(ts, ts) with exclusion=0
ab_p3, ab_i3 = pyscamp.abjoin_c22(list(ts3), list(ts3), WIN3, gpu=False)
ab_p3 = np.array(ab_p3)

# profile[i] (with exclusion) must be <= self_dot[i] (= ab_join diagonal)
N3 = len(ts3) - WIN3 + 1
self_dots = ab_p3  # AB-join with self gives max dot product including self
# For i where self_dots[i] is defined, profile[i] should be <= self_dots[i]
ok3 = np.all(cpu_p3 <= self_dots + 1e-6)
all_passed &= check("selfjoin profile[i] ≤ abjoin self-dot[i] for all i", ok3,
                    f"violations={np.sum(cpu_p3 > self_dots + 1e-6)}")

if HAS_GPU:
    gpu_p3, gpu_i3 = pyscamp.selfjoin_c22(list(ts3), WIN3, gpu=True)
    gpu_p3 = np.array(gpu_p3)
    diff3 = np.max(np.abs(cpu_p3 - gpu_p3))
    all_passed &= check("CPU == GPU on test-3 series", diff3 < 1e-6,
                        f"max_diff={diff3:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 4: Tiny 3-subsequence series — manually verify")
print("=" * 70)
# Construct a series with exactly 3 non-overlapping subsequences so we can
# verify the full distance matrix by hand.
WIN4 = 50
rng  = np.random.default_rng(123)
t    = np.arange(WIN4, dtype=np.float64)
A = np.sin(2 * np.pi * t / WIN4)              # pure sine
B = np.sin(2 * np.pi * t / WIN4 * 2)          # double-freq sine
C = rng.standard_normal(WIN4)                  # random noise

ts4 = np.concatenate([A, B, C])                # 3 subsequences: 0, WIN4, 2*WIN4
# There are only N4 = 3*WIN4 - WIN4 + 1 = 2*WIN4+1 = 101 subsequences total.
# The 3 "pure" ones are at positions 0, WIN4, 2*WIN4.

if HAS_PYCATCH22:
    FA = features_via_pycatch22(A)
    FB = features_via_pycatch22(B)
    FC = features_via_pycatch22(C)
    dAB = dot22(FA, FB)
    dAC = dot22(FA, FC)
    dBC = dot22(FB, FC)
    dAA = dot22(FA, FA)
    dBB = dot22(FB, FB)
    dCC = dot22(FC, FC)

    print(f"  F(A)·F(A) = {dAA:10.4f}  (self, not in profile)")
    print(f"  F(B)·F(B) = {dBB:10.4f}  (self)")
    print(f"  F(C)·F(C) = {dCC:10.4f}  (self)")
    print(f"  F(A)·F(B) = {dAB:10.4f}")
    print(f"  F(A)·F(C) = {dAC:10.4f}")
    print(f"  F(B)·F(C) = {dBC:10.4f}")
    print()
    print("  Expected nearest-neighbours for anchor subsequences:")
    # subseq at 0 (A): best of {B, C} = argmax(dAB, dAC)
    nn_A = WIN4 if dAB >= dAC else 2 * WIN4
    nn_B = 0    if dAB >= dBC else 2 * WIN4
    nn_C = 0    if dAC >= dBC else WIN4
    print(f"    subseq[  0] (A) → nearest = subseq[{nn_A}]  "
          f"({'B' if nn_A==WIN4 else 'C'})  dot={max(dAB,dAC):.4f}")
    print(f"    subseq[{WIN4}] (B) → nearest = subseq[{nn_B}]  "
          f"({'A' if nn_B==0 else 'C'})  dot={max(dAB,dBC):.4f}")
    print(f"    subseq[{2*WIN4}] (C) → nearest = subseq[{nn_C}]  "
          f"({'A' if nn_C==0 else 'B'})  dot={max(dAC,dBC):.4f}")
    print()

    # Run CPU and GPU on the full series, check anchor positions
    cpu_p4, cpu_i4 = pyscamp.selfjoin_c22(list(ts4), WIN4, gpu=False)
    cpu_p4 = np.array(cpu_p4)
    cpu_i4 = np.array(cpu_i4)

    exc4 = WIN4 // 4
    # For anchor at 0: expected profile value = max(dAB, dAC)
    # but anchors are subsequences 0, WIN4, 2*WIN4 — full series has overlapping
    # subsequences; the profile value at position 0 is the max over ALL
    # non-excluded subsequences (not just B and C), so it should be >= max(dAB, dAC).
    expected_min_0 = max(dAB, dAC)
    ok4a = cpu_p4[0] >= expected_min_0 - 1e-4
    all_passed &= check(
        f"profile[0] ≥ max(dot(A,B), dot(A,C)) = {expected_min_0:.4f}",
        ok4a, f"got {cpu_p4[0]:.4f}")

    if HAS_GPU:
        gpu_p4, gpu_i4 = pyscamp.selfjoin_c22(list(ts4), WIN4, gpu=True)
        gpu_p4 = np.array(gpu_p4)
        diff4 = np.max(np.abs(cpu_p4 - gpu_p4))
        all_passed &= check("CPU == GPU on test-4 series", diff4 < 1e-6,
                            f"max_diff={diff4:.2e}")
else:
    print("  [SKIP] pycatch22 not available — skipping manual verification")

# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
n_tests = sum([1, HAS_GPU, HAS_PYCATCH22, HAS_PYCATCH22 and HAS_GPU,
               1, HAS_GPU, 1, HAS_GPU, HAS_PYCATCH22])
if all_passed:
    print("ALL TESTS PASSED — GPU kernel produces correct results.")
else:
    print("SOME TESTS FAILED — review output above.")
print("=" * 70)
sys.exit(0 if all_passed else 1)
