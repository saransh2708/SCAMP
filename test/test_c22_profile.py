"""Test suite for C22 Profile (pyscamp.selfjoin_c22 / pyscamp.abjoin_c22)."""

import sys
import os
import math
import numpy as np

# Add the build directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

import pyscamp


def to_np(result):
    """Convert (profile_list, index_list) to numpy arrays."""
    p, i = result
    return np.array(p, dtype=np.float64), np.array(i, dtype=np.int32)


def test_selfjoin_basic():
    """Self-join on a simple periodic signal should produce valid output."""
    print("Test 1: Self-join basic ...", end=" ")
    ts = [math.sin(2 * math.pi * i / 50) for i in range(200)]
    m = 25

    profile, index = to_np(pyscamp.selfjoin_c22(ts, m))

    n = len(ts) - m + 1
    assert len(profile) == n, f"Expected {n} profile values, got {len(profile)}"
    assert len(index) == n, f"Expected {n} indices, got {len(index)}"

    # All profile values should be finite
    assert all(np.isfinite(profile)), "Profile contains non-finite values"

    # All indices should be valid
    assert all(0 <= idx < n for idx in index), "Index out of range"

    # No self-match: index[i] should differ from i by more than exclusion zone
    exclusion = m // 4
    for i in range(n):
        assert abs(i - index[i]) > exclusion, (
            f"Self-match at i={i}, index={index[i]}, exclusion={exclusion}")

    print("PASSED")


def test_selfjoin_periodic():
    """For a periodic signal, matches should be outside the exclusion zone."""
    print("Test 2: Self-join periodic pattern ...", end=" ")
    period = 50
    ts = [math.sin(2 * math.pi * i / period) for i in range(300)]
    m = 25
    exclusion = m // 4

    profile, index = to_np(pyscamp.selfjoin_c22(ts, m))
    n = len(ts) - m + 1

    # All matches should be outside the exclusion zone
    for i in range(n):
        assert abs(i - index[i]) > exclusion, \
            f"Exclusion violation at i={i}, index={index[i]}"

    # For a pure periodic signal, all profile values should be similar
    # (every subsequence has a near-identical C22 match elsewhere)
    profile_std = np.std(profile)
    profile_mean = np.mean(profile)
    cv = profile_std / abs(profile_mean) if abs(profile_mean) > 0 else 0
    print(f"(CV={cv:.4f}) ", end="")
    assert cv < 0.5, f"Profile values too variable for a periodic signal (CV={cv:.4f})"

    print("PASSED")


def test_abjoin_basic():
    """AB-join: querying A against B should return valid results."""
    print("Test 3: AB-join basic ...", end=" ")
    ts_a = [math.sin(2 * math.pi * i / 50) for i in range(150)]
    ts_b = [math.sin(2 * math.pi * i / 50 + 0.5) for i in range(200)]
    m = 25

    profile, index = to_np(pyscamp.abjoin_c22(ts_a, ts_b, m))

    na = len(ts_a) - m + 1
    nb = len(ts_b) - m + 1
    assert len(profile) == na, f"Expected {na} profile values, got {len(profile)}"
    assert len(index) == na

    # All profile values should be finite
    assert all(np.isfinite(profile)), "Profile contains non-finite values"

    # Indices should be valid (into B's subsequences)
    assert all(0 <= idx < nb for idx in index), "Index out of range for B"

    print("PASSED")


def test_abjoin_identical():
    """AB-join with identical series should match self-join results."""
    print("Test 4: AB-join identical series ...", end=" ")
    ts = [math.sin(2 * math.pi * i / 50) + 0.1 * (i % 7) for i in range(150)]
    m = 20

    profile_ab, index_ab = to_np(pyscamp.abjoin_c22(ts, ts, m))
    profile_self, index_self = to_np(pyscamp.selfjoin_c22(ts, m))

    # AB-join with identical series has no exclusion zone, so profile values
    # should be >= self-join values (self-join excludes nearby matches)
    for i in range(len(profile_self)):
        assert profile_ab[i] >= profile_self[i] - 1e-10, (
            f"AB-join profile[{i}]={profile_ab[i]} < self-join profile[{i}]={profile_self[i]}")

    print("PASSED")


def test_shift_invariance():
    """C22 features are z-scored, so shifting/scaling should produce similar profiles."""
    print("Test 5: Shift/scale invariance ...", end=" ")
    ts = [math.sin(2 * math.pi * i / 50) + 0.1 * (i % 7) for i in range(150)]
    ts_shifted = [x * 5 + 100 for x in ts]
    m = 25

    profile1, index1 = to_np(pyscamp.selfjoin_c22(ts, m))
    profile2, index2 = to_np(pyscamp.selfjoin_c22(ts_shifted, m))

    # Relative difference: dot products can be large so use relative tolerance
    rel_diffs = [abs(p1 - p2) / max(abs(p1), abs(p2), 1e-12)
                 for p1, p2 in zip(profile1, profile2)]
    max_rel_diff = max(rel_diffs)
    print(f"(max rel diff = {max_rel_diff:.2e}) ", end="")

    # Z-scored features should be almost identical; allow for numerical noise
    assert max_rel_diff < 0.01, \
        f"Profiles differ by {max_rel_diff:.4f} relative after shift+scale"

    print("PASSED")


def test_threading():
    """Results should be identical regardless of thread count."""
    print("Test 6: Thread consistency ...", end=" ")
    ts = [math.sin(2 * math.pi * i / 30) + 0.2 * math.cos(i) for i in range(200)]
    m = 20

    profile_1t, index_1t = to_np(pyscamp.selfjoin_c22(ts, m, threads=1))
    profile_4t, index_4t = to_np(pyscamp.selfjoin_c22(ts, m, threads=4))

    max_diff = max(abs(p1 - p2) for p1, p2 in zip(profile_1t, profile_4t))
    assert max_diff < 1e-12, f"Thread inconsistency: max diff = {max_diff}"
    assert list(index_1t) == list(index_4t), "Indices differ across thread counts"

    print("PASSED")


def test_two_class_discrimination():
    """C22 profile should discriminate between different signal types."""
    print("Test 7: Two-class discrimination ...", end=" ")
    # Time series with two distinct patterns: sine region + sawtooth region
    n = 100
    sine = [math.sin(2 * math.pi * i / 20) for i in range(n)]
    sawtooth = [(i % 20) / 20.0 for i in range(n)]
    ts = sine + sawtooth
    m = 20

    profile, index = to_np(pyscamp.selfjoin_c22(ts, m))

    # Subsequences in the sine region should match other sine subsequences
    sine_subs = n - m + 1  # number of sine-region subsequences
    sine_matches_sine = sum(1 for i in range(sine_subs)
                           if index[i] < sine_subs)
    sine_match_rate = sine_matches_sine / sine_subs

    print(f"(sine→sine match rate = {sine_match_rate:.1%}) ", end="")
    assert sine_match_rate > 0.5, \
        f"Expected sine subsequences to mostly match sine, got {sine_match_rate:.1%}"

    print("PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("C22 Profile Test Suite")
    print("=" * 60)
    print()

    test_selfjoin_basic()
    test_selfjoin_periodic()
    test_abjoin_basic()
    test_abjoin_identical()
    test_shift_invariance()
    test_threading()
    test_two_class_discrimination()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
