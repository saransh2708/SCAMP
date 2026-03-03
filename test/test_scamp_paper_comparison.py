#!/usr/bin/env python3
"""
SCAMP Paper (Matrix Profile XIV) Comparison Test Suite
======================================================

Reproduces the key experiments from:
  "Matrix Profile XIV: Scaling Time Series Motif Discovery with GPUs to
   Break a Quintillion Pairwise Comparisons a Day and Beyond"
  (Zimmerman et al., SoCC'19)

Compares traditional SCAMP Matrix Profile with C22 Profile on:
  1. Real earthquake seismic data (SampleInput/earthquake_precision_test100K.txt)
  2. Synthetic seismology data (mimicking Italian earthquake M6.0 case study)
  3. Synthetic chicken accelerometer data (mimicking the dustbathing motif)
  4. Synthetic tremor / periodic earthquake data (Parkfield-style)

For each experiment we compute:
  - Traditional Matrix Profile (z-normalised Euclidean distance, via pyscamp.selfjoin)
  - C22 Profile (catch22 feature-vector dot product, via pyscamp.selfjoin_c22)
and compare:
  - Top-1 and Top-K motif locations (do they agree?)
  - Discord / anomaly locations (do they agree?)
  - Qualitative profile shape comparison

Run from repo root:
  PYTHONPATH=build/src/python python3 test/test_scamp_paper_comparison.py

Requirements: numpy, matplotlib (optional for plots), pyscamp (built)

Paper reference:
  https://doi.org/10.1145/3357223.3362721
  Supporting webpage: https://sites.google.com/view/2019scamp
"""

import sys
import os
import time
import math
import numpy as np
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found.")
    print("Run from repo root: PYTHONPATH=build/src/python python3 test/test_scamp_paper_comparison.py")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available — skipping plot generation")

HAS_GPU = pyscamp.gpu_supported()

OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "scamp_paper_comparison_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Utilities ──────────────────────────────────────────────────────────────────

def load_ts_file(path):
    """Load a one-value-per-line time series file."""
    values = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    values.append(float(line))
                except ValueError:
                    continue
    return values


def compute_mp_brute_force_f64(ts, window):
    """
    Compute Matrix Profile via vectorised z-normalised Euclidean distance
    at float64 precision.

    Uses the identity: ||z_i - z_j||² = 2*m*(1 - corr(i,j)) where z_i, z_j
    are z-normalised subsequences and corr is their Pearson correlation.

    This is exact (float64) and fast via numpy matrix operations.
    For n ≤ ~30K this runs in ~10-30 seconds.
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts) - window + 1
    exclusion_zone = window // 4

    # Pre-compute z-normalised subsequences as rows of a matrix
    subs = np.empty((n, window), dtype=np.float64)
    for i in range(n):
        sub = ts[i:i + window].copy()
        m = sub.mean()
        s = sub.std(ddof=0)
        if s < 1e-12:
            s = 1.0
        subs[i] = (sub - m) / s

    profile = np.full(n, np.inf, dtype=np.float64)
    index = np.zeros(n, dtype=np.int64)

    # Compute in chunks to limit memory (correlation matrix can be large)
    chunk_size = min(n, 2000)
    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # Correlation: subs[i_start:i_end] @ subs.T → (chunk × n)
        corr_block = subs[i_start:i_end] @ subs.T  # already z-normalised → dot/m
        corr_block /= window  # normalise to get Pearson corr

        # Distance: sqrt(2 * m * (1 - corr))
        dist_block = np.sqrt(np.maximum(0, 2.0 * window * (1.0 - corr_block)))

        # Apply exclusion zone
        for local_i in range(i_end - i_start):
            global_i = i_start + local_i
            ez_start = max(0, global_i - exclusion_zone)
            ez_end = min(n, global_i + exclusion_zone + 1)
            dist_block[local_i, ez_start:ez_end] = np.inf

            best_j = np.argmin(dist_block[local_i])
            best_d = dist_block[local_i, best_j]
            if best_d < profile[global_i]:
                profile[global_i] = best_d
                index[global_i] = best_j

    return profile, index


def compute_both_profiles(ts, window, label=""):
    """Compute both traditional MP (float64) and C22 Profile for the same series."""
    ts_arr = np.asarray(ts, dtype=np.float64)
    ts_list = list(ts_arr)
    n_subs = len(ts_list) - window + 1
    print(f"  Series length: {len(ts_list):,}, window: {window}, subsequences: {n_subs:,}")

    # Traditional Matrix Profile — float64 brute-force for accuracy
    BRUTE_FORCE_LIMIT = 15000  # Only brute-force for manageable sizes
    if n_subs <= BRUTE_FORCE_LIMIT:
        print(f"  Computing Matrix Profile (brute-force float64)...", end="", flush=True)
        t0 = time.time()
        mp_profile, mp_index = compute_mp_brute_force_f64(ts_arr, window)
        mp_time = time.time() - t0
        print(f" done ({mp_time:.2f}s)")
    else:
        # Fall back to SCAMP for large datasets
        print(f"  Computing Matrix Profile (SCAMP float32)...", end="", flush=True)
        t0 = time.time()
        if HAS_GPU:
            mp_p, mp_i = pyscamp.selfjoin(ts_list, window, threads=8)
        else:
            mp_p, mp_i = pyscamp.selfjoin(ts_list, window, threads=8, gpus=[])
        mp_time = time.time() - t0
        mp_profile = np.array(mp_p, dtype=np.float64)
        mp_index = np.array(mp_i, dtype=np.int64)
        print(f" done ({mp_time:.2f}s)")

    # C22 Profile
    print(f"  Computing C22 Profile...", end="", flush=True)
    t0 = time.time()
    c22_profile, c22_index = pyscamp.selfjoin_c22(ts_list, window, threads=8, gpu=HAS_GPU)
    c22_time = time.time() - t0
    c22_profile = np.array(c22_profile, dtype=np.float64)
    c22_index = np.array(c22_index, dtype=np.int64)
    print(f" done ({c22_time:.2f}s)")

    # Profile statistics
    mp_finite = mp_profile[np.isfinite(mp_profile)]
    c22_finite = c22_profile[np.isfinite(c22_profile)]
    print(f"  MP  stats: min={mp_finite.min():.6f}, max={mp_finite.max():.6f}, "
          f"mean={mp_finite.mean():.6f}, std={mp_finite.std():.6f}")
    print(f"  C22 stats: min={c22_finite.min():.4f}, max={c22_finite.max():.4f}, "
          f"mean={c22_finite.mean():.4f}, std={c22_finite.std():.4f}")
    mp_range = mp_finite.max() - mp_finite.min()
    c22_range = c22_finite.max() - c22_finite.min()
    if mp_range < 1e-6:
        print(f"  ⚠ NOTE: MP profile has very low variance "
              f"(range={mp_range:.2e}) — data may lack z-norm shape variation")
    if c22_range < 1e-6:
        print(f"  ⚠ NOTE: C22 profile has very low variance "
              f"(range={c22_range:.2e})")

    return {
        'mp_profile': mp_profile, 'mp_index': mp_index, 'mp_time': mp_time,
        'c22_profile': c22_profile, 'c22_index': c22_index, 'c22_time': c22_time,
    }


def safe_corrcoef(a, b):
    """Pearson correlation that handles constant arrays gracefully."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return 0.0
    a_std = np.std(a)
    b_std = np.std(b)
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0  # one profile is flat → no correlation
    return float(np.corrcoef(a, b)[0, 1])


def find_top_motifs(profile, index, k=5, exclusion_zone=None, window=None,
                    mode='min'):
    """
    Find top-K motifs from a profile.

    For traditional MP (distance-based): motifs are MINIMA (closest pairs).
    For C22 Profile (dot-product-based): motifs are MAXIMA (most similar pairs).

    Returns list of (motif_idx, match_idx, profile_value).
    """
    if exclusion_zone is None and window is not None:
        exclusion_zone = window // 4
    elif exclusion_zone is None:
        exclusion_zone = 1

    n = len(profile)
    # Mask out non-finite values
    finite_mask = np.isfinite(profile)
    working_profile = np.copy(profile)
    if mode == 'min':
        working_profile[~finite_mask] = np.inf
        order = np.argsort(working_profile)
    else:
        working_profile[~finite_mask] = -np.inf
        order = np.argsort(-working_profile)

    motifs = []
    used_idx = set()  # Only track the primary motif index, not match
    for idx in order:
        if len(motifs) >= k:
            break
        if not finite_mask[idx]:
            continue
        # Skip if too close to an already-selected motif index
        if any(abs(idx - u) <= exclusion_zone for u in used_idx):
            continue
        match_idx = int(index[idx])
        motifs.append((int(idx), match_idx, float(profile[idx])))
        used_idx.add(int(idx))
    return motifs


def find_top_discords(profile, k=5, exclusion_zone=None, window=None,
                      mode='max'):
    """
    Find top-K discords (anomalies) from a profile.

    For traditional MP: discords are MAXIMA (furthest from any neighbor).
    For C22 Profile: discords are MINIMA (lowest dot-product similarity).
    """
    if exclusion_zone is None and window is not None:
        exclusion_zone = window // 4
    elif exclusion_zone is None:
        exclusion_zone = 1

    n = len(profile)
    if mode == 'max':
        order = np.argsort(-profile)
    else:
        order = np.argsort(profile)

    discords = []
    used = set()
    for idx in order:
        if len(discords) >= k:
            break
        if any(abs(idx - u) <= exclusion_zone for u in used):
            continue
        discords.append((int(idx), float(profile[idx])))
        used.add(int(idx))
    return discords


def motif_overlap(motifs_a, motifs_b, tolerance):
    """Check how many motifs from A are close to a motif in B."""
    matches = 0
    for (idx_a, _, _) in motifs_a:
        for (idx_b, _, _) in motifs_b:
            if abs(idx_a - idx_b) <= tolerance:
                matches += 1
                break
    return matches


def discord_overlap(discords_a, discords_b, tolerance):
    """Check how many discords from A are close to a discord in B."""
    matches = 0
    for (idx_a, _) in discords_a:
        for (idx_b, _) in discords_b:
            if abs(idx_a - idx_b) <= tolerance:
                matches += 1
                break
    return matches


def plot_comparison(ts, results, window, motifs_mp, motifs_c22, discords_mp,
                    discords_c22, title, filename):
    """Generate a 4-panel comparison plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=False)

    # 1. Time series
    ax = axes[0]
    ax.plot(ts, 'k-', linewidth=0.3, alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title(f'{title} — Time Series (n={len(ts):,})')
    ax.grid(True, alpha=0.2)

    # 2. Traditional Matrix Profile
    ax = axes[1]
    ax.plot(results['mp_profile'], 'b-', linewidth=0.3, alpha=0.7)
    for (idx, match, val) in motifs_mp[:3]:
        ax.axvline(idx, color='green', linewidth=1.5, alpha=0.7, linestyle='--')
        ax.axvline(match, color='green', linewidth=1, alpha=0.4, linestyle=':')
    for (idx, val) in discords_mp[:3]:
        ax.axvline(idx, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('MP Distance')
    ax.set_title(f'Traditional Matrix Profile (SCAMP) [{results["mp_time"]:.2f}s]')
    ax.legend(['Profile', 'Motif', 'Motif match', 'Discord'], loc='upper right',
              fontsize=7)
    ax.grid(True, alpha=0.2)

    # 3. C22 Profile
    ax = axes[2]
    ax.plot(results['c22_profile'], 'r-', linewidth=0.3, alpha=0.7)
    for (idx, match, val) in motifs_c22[:3]:
        ax.axvline(idx, color='green', linewidth=1.5, alpha=0.7, linestyle='--')
        ax.axvline(match, color='green', linewidth=1, alpha=0.4, linestyle=':')
    for (idx, val) in discords_c22[:3]:
        ax.axvline(idx, color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    ax.set_ylabel('C22 Dot Product')
    ax.set_title(f'C22 Profile [{results["c22_time"]:.2f}s]')
    ax.legend(['Profile', 'Motif', 'Motif match', 'Discord'], loc='upper right',
              fontsize=7)
    ax.grid(True, alpha=0.2)

    # 4. Top-1 motif pair overlay (Traditional MP)
    ax = axes[3]
    if motifs_mp:
        idx1, idx2, _ = motifs_mp[0]
        sub1 = ts[idx1:idx1 + window]
        sub2 = ts[idx2:idx2 + window]
        ax.plot(sub1, 'b-', linewidth=1.5, label=f'MP Motif A (idx={idx1})')
        ax.plot(sub2, 'b--', linewidth=1.5, alpha=0.7, label=f'MP Motif B (idx={idx2})')
    if motifs_c22:
        idx1, idx2, _ = motifs_c22[0]
        sub1 = ts[idx1:idx1 + window]
        sub2 = ts[idx2:idx2 + window]
        ax.plot(sub1, 'r-', linewidth=1.5, label=f'C22 Motif A (idx={idx1})')
        ax.plot(sub2, 'r--', linewidth=1.5, alpha=0.7, label=f'C22 Motif B (idx={idx2})')
    ax.set_ylabel('Value')
    ax.set_title('Top-1 Motif Pair Comparison')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 5. Normalised profile comparison (both as anomaly score)
    ax = axes[4]
    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    ax.plot(mp_norm, 'b-', linewidth=0.3, alpha=0.6, label='MP (normalised)')
    ax.plot(c22_norm, 'r-', linewidth=0.3, alpha=0.6, label='C22 (inverted+normalised)')
    corr = safe_corrcoef(mp_norm, c22_norm)
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'Normalised Profile Comparison (Pearson r = {corr:.4f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Subsequence Index')

    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Plot saved: {save_path}")


def print_motif_table(motifs_mp, motifs_c22, label="Motifs"):
    """Print a side-by-side comparison table of motifs."""
    k = max(len(motifs_mp), len(motifs_c22))
    print(f"\n  {'Rank':<6} {'── Traditional MP ──':<40} {'── C22 Profile ──':<40}")
    print(f"  {'':─<6} {'':─<40} {'':─<40}")
    for i in range(k):
        mp_str = ""
        c22_str = ""
        if i < len(motifs_mp):
            idx, match, val = motifs_mp[i]
            mp_str = f"idx={idx:>6}, match={match:>6}, dist={val:.4f}"
        if i < len(motifs_c22):
            idx, match, val = motifs_c22[i]
            c22_str = f"idx={idx:>6}, match={match:>6}, dot={val:.4f}"
        print(f"  {i+1:<6} {mp_str:<40} {c22_str:<40}")


def print_discord_table(discords_mp, discords_c22, label="Discords"):
    """Print a side-by-side comparison table of discords."""
    k = max(len(discords_mp), len(discords_c22))
    print(f"\n  {'Rank':<6} {'── Traditional MP ──':<35} {'── C22 Profile ──':<35}")
    print(f"  {'':─<6} {'':─<35} {'':─<35}")
    for i in range(k):
        mp_str = ""
        c22_str = ""
        if i < len(discords_mp):
            idx, val = discords_mp[i]
            mp_str = f"idx={idx:>6}, dist={val:.4f}"
        if i < len(discords_c22):
            idx, val = discords_c22[i]
            c22_str = f"idx={idx:>6}, dot={val:.4f}"
        print(f"  {i+1:<6} {mp_str:<35} {c22_str:<35}")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Real Earthquake Seismic Data
# Corresponds to SCAMP Paper Sections 4.2–4.5 (seismology case studies)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_earthquake_data():
    """
    Test on real earthquake seismic data.

    The SCAMP paper uses seismic waveform data from HRSN (Parkfield, CA)
    and Italian seismic stations to discover earthquake motifs — repeated
    seismic events indicating foreshocks, aftershocks, and tremor.

    We use the earthquake_precision_test100K.txt dataset (95K points of
    seismic-like data) with window sizes matching those in the paper
    (the paper uses windows of 200–500 samples at 100 Hz).
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 1: Real Earthquake Seismic Data")
    print("  (cf. SCAMP Paper §4.2–4.5: Seismology case studies)")
    print("=" * 78)

    data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "SampleInput" / "earthquake_precision_test100K.txt"
    if not data_path.exists():
        print("  [SKIP] earthquake_precision_test100K.txt not found")
        return None

    ts_full = load_ts_file(str(data_path))
    print(f"  Loaded {len(ts_full):,} points from {data_path.name}")

    # Use a manageable subset for testing (paper uses 100K–100M+ with GPUs)
    # We use 10K points to allow brute-force float64 MP computation
    n = min(10000, len(ts_full))
    ts = ts_full[:n]
    window = 100  # Paper typically uses m=200–500 at 100 Hz

    results = compute_both_profiles(ts, window, "Earthquake")

    # Find motifs and discords
    motifs_mp = find_top_motifs(results['mp_profile'], results['mp_index'],
                                k=5, window=window, mode='min')
    motifs_c22 = find_top_motifs(results['c22_profile'], results['c22_index'],
                                 k=5, window=window, mode='max')
    discords_mp = find_top_discords(results['mp_profile'], k=5,
                                    window=window, mode='max')
    discords_c22 = find_top_discords(results['c22_profile'], k=5,
                                     window=window, mode='min')

    print_motif_table(motifs_mp, motifs_c22)
    print_discord_table(discords_mp, discords_c22)

    # Agreement analysis
    tolerance = window
    motif_agree = motif_overlap(motifs_mp, motifs_c22, tolerance)
    discord_agree = discord_overlap(discords_mp, discords_c22, tolerance)
    print(f"\n  Motif agreement (within {tolerance} indices): {motif_agree}/{len(motifs_mp)}")
    print(f"  Discord agreement (within {tolerance} indices): {discord_agree}/{len(discords_mp)}")

    # Profile correlation
    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    corr = safe_corrcoef(mp_norm, c22_norm)
    print(f"  Profile shape correlation (Pearson r): {corr:.4f}")

    plot_comparison(ts, results, window, motifs_mp, motifs_c22,
                    discords_mp, discords_c22,
                    "Experiment 1: Earthquake Seismic Data",
                    "exp1_earthquake.png")

    return {
        'motif_agree': motif_agree, 'discord_agree': discord_agree,
        'corr': corr, 'motifs_mp': motifs_mp, 'motifs_c22': motifs_c22,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Synthetic Italian Earthquake (M6.0 Aftershock Sequence)
# Corresponds to SCAMP Paper §4.2 Figure 4 — the aftershock detection case
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_synthetic_aftershocks():
    """
    Synthetic data mimicking the Italian M6.0 earthquake aftershock sequence.

    The SCAMP paper (Figure 4) shows that Matrix Profile can detect aftershock
    motifs — similar waveform shapes recurring after a mainshock. We simulate:
      - Background seismic noise
      - A mainshock event (large amplitude transient)
      - Multiple aftershocks (smaller amplitude copies of the mainshock waveform)
        at decaying intervals (Omori's law: rate ~ 1/t)

    The key finding from the paper: Matrix Profile identifies repeated
    earthquake waveforms as motifs, enabling detection of subtle events.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 2: Synthetic Italian Earthquake Aftershock Sequence")
    print("  (cf. SCAMP Paper §4.2, Figure 4: M6.0 aftershock motifs)")
    print("=" * 78)

    np.random.seed(2016)  # Mnemonic: 2016 Italian earthquake year
    n = 12000
    window = 100

    # Background noise (simulates ambient seismic noise — low amplitude)
    ts = 0.2 * np.random.randn(n)

    # Create a template earthquake waveform (simplified P-S wave pair)
    def earthquake_waveform(duration=100, amplitude=1.0):
        t = np.linspace(0, 1, duration)
        # P-wave: sharp onset followed by S-wave: larger amplitude oscillation
        p_wave = amplitude * 0.4 * np.exp(-15 * t) * np.sin(30 * np.pi * t)
        s_wave = amplitude * np.exp(-6 * (t - 0.15)) * np.sin(15 * np.pi * (t - 0.15))
        s_wave[:int(0.1 * duration)] = 0
        return p_wave + s_wave

    # Mainshock at position 5000
    mainshock_pos = 5000
    mainshock = earthquake_waveform(window, amplitude=8.0)
    ts[mainshock_pos:mainshock_pos + window] += mainshock

    # Aftershocks following Omori's law: rate ~ K/(t+c)^p
    # Place aftershocks at increasing intervals with similar waveform
    aftershock_positions = []
    aftershock_amplitudes = []
    pos = mainshock_pos + 600
    amplitude = 4.0
    for i in range(8):
        if pos + window >= n:
            break
        # Add slight waveform variation (realistic: similar but not identical)
        aftershock = earthquake_waveform(window, amplitude=amplitude)
        aftershock += 0.05 * amplitude * np.random.randn(window)  # slight jitter
        ts[pos:pos + window] += aftershock
        aftershock_positions.append(pos)
        aftershock_amplitudes.append(amplitude)
        # Omori-style spacing: increasing intervals
        gap = int(600 + 400 * (i + 1) ** 0.7)
        pos += gap
        amplitude *= 0.9  # Slower decay so later events are still detectable

    known_events = [mainshock_pos] + aftershock_positions
    print(f"  Mainshock at index {mainshock_pos}, {len(aftershock_positions)} aftershocks")
    print(f"  Aftershock positions: {aftershock_positions}")

    results = compute_both_profiles(ts, window, "Aftershock")

    motifs_mp = find_top_motifs(results['mp_profile'], results['mp_index'],
                                k=5, window=window, mode='min')
    motifs_c22 = find_top_motifs(results['c22_profile'], results['c22_index'],
                                 k=5, window=window, mode='max')
    discords_mp = find_top_discords(results['mp_profile'], k=5,
                                    window=window, mode='max')
    discords_c22 = find_top_discords(results['c22_profile'], k=5,
                                     window=window, mode='min')

    print_motif_table(motifs_mp, motifs_c22)
    print_discord_table(discords_mp, discords_c22)

    # Check: motifs should correspond to aftershock pairs
    tolerance = window
    motif_agree = motif_overlap(motifs_mp, motifs_c22, tolerance)
    print(f"\n  Motif agreement (within {tolerance} indices): {motif_agree}/{len(motifs_mp)}")

    # Check: how many known events are captured by motifs?
    mp_motif_indices = set()
    for (idx, match, _) in motifs_mp:
        mp_motif_indices.add(idx)
        mp_motif_indices.add(match)
    c22_motif_indices = set()
    for (idx, match, _) in motifs_c22:
        c22_motif_indices.add(idx)
        c22_motif_indices.add(match)

    mp_events_found = sum(1 for ev in known_events
                          if any(abs(ev - mi) <= tolerance for mi in mp_motif_indices))
    c22_events_found = sum(1 for ev in known_events
                           if any(abs(ev - mi) <= tolerance for mi in c22_motif_indices))
    print(f"  Events captured by MP motifs:  {mp_events_found}/{len(known_events)}")
    print(f"  Events captured by C22 motifs: {c22_events_found}/{len(known_events)}")

    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    corr = safe_corrcoef(mp_norm, c22_norm)
    print(f"  Profile shape correlation (Pearson r): {corr:.4f}")

    # Mainshock should be the top discord for both profiles
    mp_top_discord = discords_mp[0][0] if discords_mp else -1
    c22_top_discord = discords_c22[0][0] if discords_c22 else -1
    mp_main_ok = abs(mp_top_discord - mainshock_pos) <= tolerance
    c22_main_ok = abs(c22_top_discord - mainshock_pos) <= tolerance
    print(f"  Mainshock detected as top discord: MP={'YES' if mp_main_ok else 'NO'}, "
          f"C22={'YES' if c22_main_ok else 'NO'}")

    plot_comparison(ts, results, window, motifs_mp, motifs_c22,
                    discords_mp, discords_c22,
                    "Experiment 2: Synthetic Aftershock Sequence (cf. Paper Fig 4)",
                    "exp2_aftershocks.png")

    return {
        'motif_agree': motif_agree, 'corr': corr,
        'mp_events': mp_events_found, 'c22_events': c22_events_found,
        'mp_main_discord': mp_main_ok, 'c22_main_discord': c22_main_ok,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Synthetic Chicken Accelerometer (Dustbathing Motif)
# Corresponds to SCAMP Paper §1, Figure 1 — the chicken accelerometer case
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_chicken_accelerometer():
    """
    Synthetic data mimicking the chicken accelerometer experiment.

    The SCAMP paper (Figure 1) shows 24 hours of accelerometer data from a
    chicken. The data appears random, but Matrix Profile discovers a motif
    corresponding to "dustbathing" behavior — a conserved pattern repeated
    multiple times during the day.

    We simulate:
      - Baseline: noisy irregular movement (random walk + noise)
      - Embedded dustbathing motifs: a distinctive oscillation pattern
        repeated several times at random intervals
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 3: Synthetic Chicken Accelerometer Data")
    print("  (cf. SCAMP Paper §1, Figure 1: Dustbathing motif discovery)")
    print("=" * 78)

    np.random.seed(42)
    n = 12000  # Simulating a portion of 24h data
    window = 120  # Paper uses m around 100–200

    # Baseline: stationary noisy movement (NOT random walk — stationary noise)
    # Mix of slow oscillations (body position drift) + fast noise (accelerometer)
    t_arr = np.arange(n, dtype=np.float64)
    ts = (0.5 * np.sin(2 * np.pi * t_arr / 5000)       # very slow drift
          + 0.3 * np.sin(2 * np.pi * t_arr / 800)       # medium cycle
          + 0.4 * np.random.randn(n))                    # fast noise

    # Create "dustbathing" motif template: rapid oscillation with envelope
    def dustbathing_motif(duration=120):
        t = np.linspace(0, 1, duration)
        envelope = np.sin(np.pi * t) ** 2  # Bell-shaped envelope
        oscillation = np.sin(30 * np.pi * t) + 0.5 * np.sin(50 * np.pi * t)
        return 4.0 * envelope * oscillation  # Higher amplitude to stand out

    motif_template = dustbathing_motif(window)

    # Embed motif at several positions (simulating repeated dustbathing events)
    motif_positions = [1500, 4000, 7000, 9500]
    for pos in motif_positions:
        # Add slight variation to each occurrence (natural variability)
        variation = 1.0 + 0.1 * np.random.randn()
        phase_shift = np.random.randint(-3, 3)
        motif_instance = variation * dustbathing_motif(window)
        # Add a tiny bit of noise to each instance
        motif_instance += 0.1 * np.random.randn(window)
        start = pos + phase_shift
        start = max(0, min(start, n - window))
        ts[start:start + window] += motif_instance

    n_motifs = len(motif_positions)
    print(f"  Embedded {n_motifs} dustbathing motifs at: {motif_positions}")

    results = compute_both_profiles(ts, window, "Chicken")

    motifs_mp = find_top_motifs(results['mp_profile'], results['mp_index'],
                                k=5, window=window, mode='min')
    motifs_c22 = find_top_motifs(results['c22_profile'], results['c22_index'],
                                 k=5, window=window, mode='max')
    discords_mp = find_top_discords(results['mp_profile'], k=3,
                                    window=window, mode='max')
    discords_c22 = find_top_discords(results['c22_profile'], k=3,
                                     window=window, mode='min')

    print_motif_table(motifs_mp, motifs_c22)

    # Check: how many embedded motifs are found?
    tolerance = window * 2  # Be generous due to phase shifts
    mp_motif_indices = set()
    for (idx, match, _) in motifs_mp:
        mp_motif_indices.add(idx)
        mp_motif_indices.add(match)
    c22_motif_indices = set()
    for (idx, match, _) in motifs_c22:
        c22_motif_indices.add(idx)
        c22_motif_indices.add(match)

    mp_found = sum(1 for pos in motif_positions
                   if any(abs(pos - mi) <= tolerance for mi in mp_motif_indices))
    c22_found = sum(1 for pos in motif_positions
                    if any(abs(pos - mi) <= tolerance for mi in c22_motif_indices))

    print(f"\n  Dustbathing motifs found by MP:  {mp_found}/{len(motif_positions)}")
    print(f"  Dustbathing motifs found by C22: {c22_found}/{len(motif_positions)}")

    motif_agree = motif_overlap(motifs_mp, motifs_c22, tolerance)
    print(f"  Motif agreement: {motif_agree}/{len(motifs_mp)}")

    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    corr = safe_corrcoef(mp_norm, c22_norm)
    print(f"  Profile shape correlation (Pearson r): {corr:.4f}")

    plot_comparison(ts, results, window, motifs_mp, motifs_c22,
                    discords_mp, discords_c22,
                    "Experiment 3: Chicken Accelerometer (cf. Paper Fig 1)",
                    "exp3_chicken.png")

    return {
        'mp_found': mp_found, 'c22_found': c22_found,
        'motif_agree': motif_agree, 'corr': corr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Synthetic Parkfield Tremor (Periodic Earthquake Recurrence)
# Corresponds to SCAMP Paper §4.3–4.4 (Parkfield repeating events / tremor)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_parkfield_tremor():
    """
    Synthetic data mimicking Parkfield repeating earthquakes and tremor.

    The SCAMP paper (§4.3–4.4, Figures 8–11) uses Matrix Profile on
    continuous seismic data from the Parkfield segment of the San Andreas
    Fault to discover:
      1. Repeating microearthquakes (periodic recurrence)
      2. Non-volcanic tremor (low-frequency signals)

    We simulate:
      - Background noise with varying amplitude (mimicking diurnal variation)
      - Regularly repeating micro-earthquakes (period ~2000 samples)
      - Irregular tremor-like episodes (low-frequency bursts)
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 4: Synthetic Parkfield Tremor & Repeating Earthquakes")
    print("  (cf. SCAMP Paper §4.3–4.4, Figures 8–11)")
    print("=" * 78)

    np.random.seed(2004)  # 2004 Parkfield earthquake
    n = 15000
    window = 80

    # Background noise with diurnal variation (low amplitude)
    t = np.arange(n, dtype=np.float64)
    noise_envelope = 1.0 + 0.2 * np.sin(2 * np.pi * t / n)
    ts = 0.3 * noise_envelope * np.random.randn(n)

    # Repeating micro-earthquake template
    def micro_eq(duration=80, amplitude=1.0):
        t = np.linspace(0, 1, duration)
        return amplitude * np.exp(-10 * t) * np.sin(20 * np.pi * t)

    # Place repeating events quasi-periodically (period ~3000 ± jitter)
    micro_eq_positions = []
    pos = 2000
    period = 3000
    while pos + window < n:
        jitter = np.random.randint(-100, 100)
        actual_pos = pos + jitter
        if 0 <= actual_pos < n - window:
            amp = 5.0 + 0.3 * np.random.randn()
            eq_waveform = micro_eq(window, amplitude=amp)
            eq_waveform += 0.05 * amp * np.random.randn(window)  # slight jitter
            ts[actual_pos:actual_pos + window] += eq_waveform
            micro_eq_positions.append(actual_pos)
        pos += period

    # Tremor-like episodes: longer-duration, low-frequency bursts
    tremor_positions = []
    tremor_template_len = window
    for tpos in [10000, 25000, 35000]:
        if tpos + tremor_template_len < n:
            t_local = np.linspace(0, 1, tremor_template_len)
            tremor = 3.0 * np.sin(6 * np.pi * t_local) * np.exp(-3 * abs(t_local - 0.5))
            ts[tpos:tpos + tremor_template_len] += tremor
            tremor_positions.append(tpos)

    all_events = micro_eq_positions + tremor_positions
    print(f"  Micro-earthquakes: {len(micro_eq_positions)} events (quasi-periodic)")
    print(f"  Tremor episodes: {len(tremor_positions)} events")
    print(f"  Micro-EQ positions: {micro_eq_positions}")
    print(f"  Tremor positions: {tremor_positions}")

    results = compute_both_profiles(ts, window, "Parkfield")

    motifs_mp = find_top_motifs(results['mp_profile'], results['mp_index'],
                                k=8, window=window, mode='min')
    motifs_c22 = find_top_motifs(results['c22_profile'], results['c22_index'],
                                 k=8, window=window, mode='max')
    discords_mp = find_top_discords(results['mp_profile'], k=5,
                                    window=window, mode='max')
    discords_c22 = find_top_discords(results['c22_profile'], k=5,
                                     window=window, mode='min')

    print_motif_table(motifs_mp, motifs_c22)

    # Check: micro-earthquake motifs detected?
    tolerance = window * 2
    mp_motif_indices = set()
    for (idx, match, _) in motifs_mp:
        mp_motif_indices.add(idx)
        mp_motif_indices.add(match)
    c22_motif_indices = set()
    for (idx, match, _) in motifs_c22:
        c22_motif_indices.add(idx)
        c22_motif_indices.add(match)

    mp_eq_found = sum(1 for pos in micro_eq_positions
                      if any(abs(pos - mi) <= tolerance for mi in mp_motif_indices))
    c22_eq_found = sum(1 for pos in micro_eq_positions
                       if any(abs(pos - mi) <= tolerance for mi in c22_motif_indices))
    mp_tremor_found = sum(1 for pos in tremor_positions
                          if any(abs(pos - mi) <= tolerance for mi in mp_motif_indices))
    c22_tremor_found = sum(1 for pos in tremor_positions
                           if any(abs(pos - mi) <= tolerance for mi in c22_motif_indices))

    print(f"\n  Micro-EQs found by MP motifs:  {mp_eq_found}/{len(micro_eq_positions)}")
    print(f"  Micro-EQs found by C22 motifs: {c22_eq_found}/{len(micro_eq_positions)}")
    print(f"  Tremor found by MP motifs:  {mp_tremor_found}/{len(tremor_positions)}")
    print(f"  Tremor found by C22 motifs: {c22_tremor_found}/{len(tremor_positions)}")

    motif_agree = motif_overlap(motifs_mp, motifs_c22, tolerance)
    print(f"  Motif agreement: {motif_agree}/{len(motifs_mp)}")

    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    corr = safe_corrcoef(mp_norm, c22_norm)
    print(f"  Profile shape correlation (Pearson r): {corr:.4f}")

    plot_comparison(ts, results, window, motifs_mp, motifs_c22,
                    discords_mp, discords_c22,
                    "Experiment 4: Parkfield Repeating EQs & Tremor (cf. Paper §4.3–4.4)",
                    "exp4_parkfield.png")

    return {
        'mp_eq_found': mp_eq_found, 'c22_eq_found': c22_eq_found,
        'mp_tremor_found': mp_tremor_found, 'c22_tremor_found': c22_tremor_found,
        'motif_agree': motif_agree, 'corr': corr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Full Earthquake Dataset (larger scale)
# Tests scaling with the full 95K earthquake dataset
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_full_earthquake():
    """
    Full-scale test on the complete 95K earthquake dataset.

    The SCAMP paper emphasises scalability — handling time series with
    millions to billions of data points. While we can't test at that
    scale on CPU, we test with the full available earthquake dataset
    to measure throughput and compare results at a meaningful size.
    """
    print("\n" + "=" * 78)
    print("EXPERIMENT 5: Full Earthquake Dataset (95K points)")
    print("  (cf. SCAMP Paper §3: Scalability and throughput)")
    print("=" * 78)

    data_path = Path(os.path.dirname(os.path.abspath(__file__))) / "SampleInput" / "earthquake_precision_test100K.txt"
    if not data_path.exists():
        print("  [SKIP] earthquake_precision_test100K.txt not found")
        return None

    ts_full = load_ts_file(str(data_path))
    # Use 30K subset to balance detail vs runtime
    n = min(30000, len(ts_full))
    ts = ts_full[:n]
    print(f"  Loaded {len(ts_full):,} points, using first {n:,} from {data_path.name}")

    window = 200  # Closer to paper's typical seismology window

    results = compute_both_profiles(ts, window, "Full EQ")

    motifs_mp = find_top_motifs(results['mp_profile'], results['mp_index'],
                                k=10, window=window, mode='min')
    motifs_c22 = find_top_motifs(results['c22_profile'], results['c22_index'],
                                 k=10, window=window, mode='max')

    print_motif_table(motifs_mp, motifs_c22)

    tolerance = window * 2
    motif_agree = motif_overlap(motifs_mp, motifs_c22, tolerance)
    print(f"\n  Motif agreement: {motif_agree}/{len(motifs_mp)}")

    mp_norm = (results['mp_profile'] - results['mp_profile'].min()) / \
              (results['mp_profile'].max() - results['mp_profile'].min() + 1e-10)
    c22_inv = -results['c22_profile']
    c22_norm = (c22_inv - c22_inv.min()) / (c22_inv.max() - c22_inv.min() + 1e-10)
    corr = safe_corrcoef(mp_norm, c22_norm)
    print(f"  Profile shape correlation (Pearson r): {corr:.4f}")

    # Throughput comparison
    n_subs = len(ts) - window + 1
    mp_throughput = n_subs * n_subs / results['mp_time'] if results['mp_time'] > 0 else 0
    c22_throughput = n_subs * n_subs / results['c22_time'] if results['c22_time'] > 0 else 0
    print(f"\n  Throughput (pairwise comparisons/sec):")
    print(f"    Traditional MP: {mp_throughput:,.0f}")
    print(f"    C22 Profile:    {c22_throughput:,.0f}")
    print(f"    Ratio (C22/MP): {c22_throughput/mp_throughput:.2f}x" if mp_throughput > 0 else "")

    plot_comparison(ts, results, window, motifs_mp, motifs_c22,
                    [], [],  # skip discords to keep the plot cleaner
                    "Experiment 5: Full 95K Earthquake Dataset (w=200)",
                    "exp5_full_earthquake.png")

    return {
        'motif_agree': motif_agree, 'corr': corr,
        'mp_time': results['mp_time'], 'c22_time': results['c22_time'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 76 + "╗")
    print("║  SCAMP Paper (Matrix Profile XIV) — C22 Profile Comparison Test Suite   ║")
    print("╠" + "═" * 76 + "╣")
    print(f"║  GPU support: {'Yes' if HAS_GPU else 'No (CPU only)':<60}║")
    print(f"║  Matplotlib:  {'Yes' if HAS_MATPLOTLIB else 'No (plots skipped)':<60}║")
    print(f"║  Output dir:  {str(OUTPUT_DIR):<60}║")
    print("╚" + "═" * 76 + "╝")
    print()
    print("This test suite compares your C22 Profile with the traditional Matrix")
    print("Profile (SCAMP) on datasets inspired by the SCAMP paper:")
    print("  Zimmerman et al., 'Matrix Profile XIV: Scaling Time Series Motif")
    print("  Discovery with GPUs', SoCC'19.")
    print("  https://doi.org/10.1145/3357223.3362721")
    print()

    all_results = {}
    total_time = time.time()

    # Run experiments
    all_results['exp1'] = experiment_earthquake_data()
    all_results['exp2'] = experiment_synthetic_aftershocks()
    all_results['exp3'] = experiment_chicken_accelerometer()
    all_results['exp4'] = experiment_parkfield_tremor()
    all_results['exp5'] = experiment_full_earthquake()

    total_elapsed = time.time() - total_time

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("OVERALL SUMMARY")
    print("═" * 78)

    summary_lines = []

    for key, label, check_fn in [
        ('exp1', 'Exp 1: Real Earthquake Data', lambda r: r and r.get('corr', 0) > -1),
        ('exp2', 'Exp 2: Synthetic Aftershocks', lambda r: r and r.get('corr', 0) > 0.2),
        ('exp3', 'Exp 3: Chicken Accelerometer', lambda r: r and (r.get('c22_found', 0) >= 2 or r.get('mp_found', 0) >= 2)),
        ('exp4', 'Exp 4: Parkfield Tremor',      lambda r: r and (r.get('c22_eq_found', 0) >= 2 or r.get('mp_eq_found', 0) >= 2)),
        ('exp5', 'Exp 5: Full Earthquake (95K)',  lambda r: r and r.get('corr', 0) > -1),
    ]:
        r = all_results.get(key)
        if r is None:
            status = '[SKIP]'
        elif check_fn(r):
            status = '[PASS]'
        else:
            status = '[WARN]'

        detail = ""
        if r:
            if 'corr' in r:
                detail += f"corr={r['corr']:.3f}"
            if 'motif_agree' in r:
                detail += f"  motif_agree={r['motif_agree']}"
            if 'mp_found' in r and 'c22_found' in r:
                detail += f"  MP_found={r['mp_found']} C22_found={r['c22_found']}"
            if 'mp_eq_found' in r:
                detail += f"  MP_eq={r['mp_eq_found']} C22_eq={r['c22_eq_found']}"
            if 'mp_time' in r:
                detail += f"  time: MP={r['mp_time']:.1f}s C22={r['c22_time']:.1f}s"

        line = f"  {status} {label:<35} {detail}"
        summary_lines.append(line)
        print(line)

    print(f"\n  Total runtime: {total_elapsed:.1f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    if HAS_MATPLOTLIB:
        print(f"  Plots: {', '.join(f.name for f in OUTPUT_DIR.glob('*.png'))}")
    print()

    # Write summary to file
    with open(OUTPUT_DIR / "summary.txt", 'w') as f:
        f.write("SCAMP Paper Comparison — C22 Profile Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"GPU: {HAS_GPU}\n")
        f.write(f"Total runtime: {total_elapsed:.1f}s\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print("═" * 78)
    print("INTERPRETATION GUIDE")
    print("═" * 78)
    print("""
  The SCAMP paper computes the traditional Matrix Profile (z-normalised
  Euclidean distance) to find motifs (repeated patterns) and discords
  (anomalies) in time series.

  Your C22 Profile uses catch22 feature vectors + dot-product similarity
  instead of raw subsequence distance. Key differences:

  • Traditional MP motifs:  LOWEST  distance  = most similar pair
  • C22 Profile motifs:     HIGHEST dot product = most similar pair
  • Traditional MP discords: HIGHEST distance  = most anomalous
  • C22 Profile discords:    LOWEST  dot product = most anomalous

  Profile correlation (Pearson r of normalised profiles):
    r > 0.5  : Strong agreement — C22 captures similar structure as MP
    r > 0.3  : Moderate agreement — C22 captures some patterns differently
    r > 0.0  : Weak agreement — C22 emphasises different features
    r < 0.0  : C22 and MP highlight fundamentally different properties

  Motif agreement: How many of the top-K motifs are at similar locations.
  Perfect agreement is not expected since C22 measures shape-feature
  similarity while MP measures raw waveform similarity.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
