#!/usr/bin/env python3
"""
Plot comprehensive benchmark results from benchmark_20papers_full.json.
Generates PPT-style charts (slides 9-11 equivalent) using real 20Papers data.

Run on LOCAL machine (no GPU/pyscamp needed — only matplotlib + numpy):
  python3 test/plot_20papers_results.py [-i benchmark_20papers_full.json] [-o output_dir/]
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("ERROR: matplotlib required.  pip install matplotlib")
    sys.exit(1)

# ─── Style defaults ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})
COLOR_C22 = '#d62728'
COLOR_MP = '#1f77b4'
COLOR_GPU = '#ff7f0e'
COLOR_CPU = '#2ca02c'


def load_results(path):
    if path == '-' or path == '/dev/stdin':
        return json.load(sys.stdin)
    with open(path, 'r') as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Plot 1: Performance comparison — CPU vs GPU for C22 and MP (slide 9a)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cpu_gpu_times(data, out):
    """Bar chart: CPU vs GPU runtime for each dataset, for both C22 and MP."""
    ds = data['per_dataset']
    gpu_enabled = data['config'].get('gpu_enabled', False)

    valid = [d for d in ds if 'time_s' in d.get('c22_cpu', {})]
    if not valid:
        print("  SKIP cpu_gpu_times: no valid data")
        return

    valid.sort(key=lambda d: d['series_length'])
    names = [d['dataset'] for d in valid]
    c22_cpu = [d['c22_cpu']['time_s'] for d in valid]
    mp_cpu = [d['mp_cpu']['time_s'] if 'time_s' in d.get('mp_cpu', {}) else 0 for d in valid]

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(names) * 0.7), 12))
    x = np.arange(len(names))
    w = 0.35

    # ── Top: C22 CPU vs GPU ──
    ax = axes[0]
    ax.bar(x - w / 2, c22_cpu, w, label='C22 CPU', color=COLOR_CPU, alpha=0.8)
    if gpu_enabled:
        c22_gpu = [d.get('c22_gpu', {}).get('time_s', 0) for d in valid]
        ax.bar(x + w / 2, c22_gpu, w, label='C22 GPU', color=COLOR_GPU, alpha=0.8)
    ax.set_ylabel('Time (s)')
    ax.set_title('C22 Profile: CPU vs GPU Runtime per Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    # ── Bottom: MP CPU vs GPU ──
    ax = axes[1]
    ax.bar(x - w / 2, mp_cpu, w, label='MP CPU', color=COLOR_CPU, alpha=0.8)
    if gpu_enabled:
        mp_gpu = [d.get('mp_gpu', {}).get('time_s', 0) for d in valid]
        ax.bar(x + w / 2, mp_gpu, w, label='MP GPU', color=COLOR_GPU, alpha=0.8)
    ax.set_ylabel('Time (s)')
    ax.set_title('Matrix Profile (SCAMP): CPU vs GPU Runtime per Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out, 'slide9a_cpu_gpu_runtime.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 2: Throughput comparison — C22 vs MP (slide 9b)
# ═════════════════════════════════════════════════════════════════════════════

def plot_throughput_comparison(data, out):
    """C22 vs MP throughput on each dataset (grouped bar)."""
    ds = data['per_dataset']
    valid = [d for d in ds
             if 'throughput_mpairs_s' in d.get('c22_cpu', {})
             and 'throughput_mpairs_s' in d.get('mp_cpu', {})]
    if not valid:
        print("  SKIP throughput_comparison: no valid data")
        return

    valid.sort(key=lambda d: d['series_length'])
    names = [d['dataset'] for d in valid]
    c22_tp = [d['c22_cpu']['throughput_mpairs_s'] for d in valid]
    mp_tp = [d['mp_cpu']['throughput_mpairs_s'] for d in valid]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.7), 6))
    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w / 2, c22_tp, w, label='C22 Profile', color=COLOR_C22, alpha=0.85)
    ax.bar(x + w / 2, mp_tp, w, label='Matrix Profile', color=COLOR_MP, alpha=0.85)
    ax.set_ylabel('Throughput (M pairs/s)', fontsize=12)
    ax.set_xlabel('Dataset (sorted by series length)', fontsize=12)
    ax.set_title('C22 vs Matrix Profile — CPU Throughput on 20Papers Datasets',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(out, 'slide9b_throughput_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 3: GPU speedup per dataset (slide 9c)
# ═════════════════════════════════════════════════════════════════════════════

def plot_gpu_speedup(data, out):
    """GPU speedup bars for C22 and MP per dataset."""
    ds = data['per_dataset']
    valid = [d for d in ds if 'c22_gpu_speedup' in d]
    if not valid:
        print("  SKIP gpu_speedup: no GPU data")
        return

    valid.sort(key=lambda d: d['series_length'])
    names = [d['dataset'] for d in valid]
    c22_su = [d['c22_gpu_speedup'] for d in valid]
    mp_su = [d.get('mp_gpu_speedup', 0) for d in valid]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.7), 6))
    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w / 2, c22_su, w, label='C22 GPU Speedup', color=COLOR_C22, alpha=0.85)
    ax.bar(x + w / 2, mp_su, w, label='MP GPU Speedup', color=COLOR_MP, alpha=0.85)
    ax.axhline(y=1.0, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_ylabel('GPU Speedup (x)', fontsize=12)
    ax.set_xlabel('Dataset (sorted by series length)', fontsize=12)
    ax.set_title('GPU Speedup over CPU — C22 vs Matrix Profile',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.legend(fontsize=11)

    for i, (cs, ms) in enumerate(zip(c22_su, mp_su)):
        ax.text(i - w / 2, cs + 0.05, f'{cs:.1f}x', ha='center', fontsize=7, color=COLOR_C22)
        ax.text(i + w / 2, ms + 0.05, f'{ms:.1f}x', ha='center', fontsize=7, color=COLOR_MP)

    plt.tight_layout()
    path = os.path.join(out, 'slide9c_gpu_speedup.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 4: Scaling by series length (slide 9d)
# ═════════════════════════════════════════════════════════════════════════════

def plot_scaling_by_size(data, out):
    """Runtime & throughput vs series length scatter (log-log)."""
    sbs = data.get('scaling_by_size')
    if not sbs or not sbs.get('sizes'):
        print("  SKIP scaling_by_size: no data")
        return

    sizes = sbs['sizes']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Runtime vs N
    ax = axes[0]
    ax.scatter(sizes, sbs['c22_cpu_times'], label='C22 CPU', color=COLOR_C22, s=50, zorder=3)
    ax.scatter(sizes, sbs['mp_cpu_times'], label='MP CPU', color=COLOR_MP, s=50, zorder=3)
    if 'c22_gpu_times' in sbs:
        gpu_t = [(s, t) for s, t in zip(sizes, sbs['c22_gpu_times']) if t is not None]
        if gpu_t:
            ax.scatter(*zip(*gpu_t), label='C22 GPU', color=COLOR_C22, marker='D', s=50,
                       edgecolors='black', lw=0.5, zorder=3)
    if 'mp_gpu_times' in sbs:
        gpu_t = [(s, t) for s, t in zip(sizes, sbs['mp_gpu_times']) if t is not None]
        if gpu_t:
            ax.scatter(*zip(*gpu_t), label='MP GPU', color=COLOR_MP, marker='D', s=50,
                       edgecolors='black', lw=0.5, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Series Length (N)', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Runtime vs Series Length (20Papers)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Throughput vs N
    ax = axes[1]
    ax.scatter(sizes, sbs['c22_cpu_throughputs'], label='C22 CPU',
               color=COLOR_C22, s=50, zorder=3)
    ax.scatter(sizes, sbs['mp_cpu_throughputs'], label='MP CPU',
               color=COLOR_MP, s=50, zorder=3)
    if 'c22_gpu_throughputs' in sbs:
        gpu_t = [(s, t) for s, t in zip(sizes, sbs['c22_gpu_throughputs']) if t is not None]
        if gpu_t:
            ax.scatter(*zip(*gpu_t), label='C22 GPU', color=COLOR_C22, marker='D', s=50,
                       edgecolors='black', lw=0.5, zorder=3)
    if 'mp_gpu_throughputs' in sbs:
        gpu_t = [(s, t) for s, t in zip(sizes, sbs['mp_gpu_throughputs']) if t is not None]
        if gpu_t:
            ax.scatter(*zip(*gpu_t), label='MP GPU', color=COLOR_MP, marker='D', s=50,
                       edgecolors='black', lw=0.5, zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Series Length (N)', fontsize=12)
    ax.set_ylabel('Throughput (M pairs/s)', fontsize=12)
    ax.set_title('Throughput vs Series Length (20Papers)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out, 'slide9d_scaling_by_size.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 5: Thread scaling (slide 9e)
# ═════════════════════════════════════════════════════════════════════════════

def plot_thread_scaling(data, out):
    """Thread scaling: speedup + throughput bar."""
    ts_data = data.get('thread_scaling')
    if not ts_data or not ts_data.get('thread_counts'):
        print("  SKIP thread_scaling: no data")
        return

    tc = ts_data['thread_counts']
    speedups = ts_data['c22_speedups']
    throughputs = ts_data['c22_throughputs']
    ds_name = ts_data['dataset']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup
    ax = axes[0]
    ax.plot(tc, speedups, 'ro-', lw=2, ms=8, label='Actual')
    ax.plot(tc, tc, 'k--', lw=1, alpha=0.5, label='Ideal (linear)')
    ax.set_xlabel('Threads', fontsize=12)
    ax.set_ylabel('Speedup (x)', fontsize=12)
    ax.set_title(f'C22 Thread Scaling — {ds_name}\n(N={ts_data["series_length"]}, '
                 f'w={ts_data["window_size"]})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Throughput bars
    ax = axes[1]
    x_pos = np.arange(len(tc))
    bars = ax.bar(x_pos, throughputs, color='coral', edgecolor='darkred', lw=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t) for t in tc])
    ax.set_xlabel('Threads', fontsize=12)
    ax.set_ylabel('Throughput (M pairs/s)', fontsize=12)
    ax.set_title('C22 CPU Throughput by Thread Count', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # GPU annotation
    if 'gpu_throughput' in ts_data and ts_data['gpu_throughput'] is not None:
        ax.axhline(ts_data['gpu_throughput'], color=COLOR_GPU, ls='--', lw=2,
                   label=f'GPU: {ts_data["gpu_throughput"]:.0f} M/s')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(out, 'slide9e_thread_scaling.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 6: Profile correlation — C22 vs MP (slide 10)
# ═════════════════════════════════════════════════════════════════════════════

def plot_profile_correlation(data, out):
    """Bar chart of Pearson r between normalized C22 and MP profiles."""
    ds = data['per_dataset']
    valid = [d for d in ds if 'profile_correlation' in d]
    if not valid:
        print("  SKIP profile_correlation: no data")
        return

    valid.sort(key=lambda d: d['profile_correlation'])
    names = [d['dataset'] for d in valid]
    corrs = [d['profile_correlation'] for d in valid]

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.6), 6))
    x = np.arange(len(names))
    colors = ['green' if c >= 0.5 else ('gold' if c >= 0.2 else 'red') for c in corrs]

    bars = ax.bar(x, corrs, color=colors, alpha=0.8, edgecolor='gray', lw=0.5)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title('C22 vs Matrix Profile — Normalized Profile Correlation',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)

    for bar, val in zip(bars, corrs):
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f'{val:.2f}', ha='center', va=va, fontsize=7, fontweight='bold')

    mean_corr = np.mean(corrs)
    ax.axhline(y=mean_corr, color='blue', ls='--', lw=1.5, alpha=0.7,
               label=f'Mean r = {mean_corr:.3f}')
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(out, 'slide10_profile_correlation.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 7: Anomaly detection precision (slide 11)
# ═════════════════════════════════════════════════════════════════════════════

def plot_anomaly_detection(data, out):
    """Grouped bar: MP vs C22 anomaly detection precision on labeled datasets."""
    ds = data['per_dataset']
    valid = [d for d in ds if d.get('anomaly_detection', {}).get('has_labels')]
    if not valid:
        print("  SKIP anomaly_detection: no labeled datasets")
        return

    valid.sort(key=lambda d: d['series_length'])
    names = [d['dataset'] for d in valid]
    mp_prec = [d['anomaly_detection']['mp_precision_at_k'] for d in valid]
    c22_prec = [d['anomaly_detection']['c22_precision_at_k'] for d in valid]
    top_k = valid[0]['anomaly_detection']['top_k']

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.7), 6))
    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w / 2, mp_prec, w, label='Matrix Profile', color=COLOR_MP, alpha=0.85)
    ax.bar(x + w / 2, c22_prec, w, label='C22 Profile', color=COLOR_C22, alpha=0.85)
    ax.set_ylabel(f'Precision @ {top_k}', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_title(f'Anomaly Detection — Top-{top_k} Discord Precision (20Papers)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)

    for i, (mp, c22) in enumerate(zip(mp_prec, c22_prec)):
        ax.text(i - w / 2, mp + 0.02, f'{mp:.0%}', ha='center', fontsize=7)
        ax.text(i + w / 2, c22 + 0.02, f'{c22:.0%}', ha='center', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out, 'slide11_anomaly_detection.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 8: Motif pair waveform correlation (slide 11b)
# ═════════════════════════════════════════════════════════════════════════════

def plot_motif_pair_correlation(data, out):
    """Scatter plot: C22 motif pair Pearson r vs MP motif pair Pearson r."""
    ds = data['per_dataset']
    valid = [d for d in ds if 'motif_info' in d]
    if not valid:
        print("  SKIP motif_pair_correlation: no motif info")
        return

    names = [d['dataset'] for d in valid]
    mp_r = [d['motif_info']['mp_motif_pair_pearson_r'] for d in valid]
    c22_r = [d['motif_info']['c22_motif_pair_pearson_r'] for d in valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: C22 r vs MP r
    ax = axes[0]
    ax.scatter(mp_r, c22_r, s=60, color='purple', alpha=0.7, edgecolors='k', lw=0.5, zorder=3)
    ax.plot([-1, 1], [-1, 1], 'k--', lw=1, alpha=0.3)
    ax.set_xlabel('MP Top Motif Pair Pearson r', fontsize=12)
    ax.set_ylabel('C22 Top Motif Pair Pearson r', fontsize=12)
    ax.set_title('Motif Pair Waveform Correlation:\nC22 vs Matrix Profile',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    for i, name in enumerate(names):
        ax.annotate(name[:15], (mp_r[i], c22_r[i]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

    # Grouped bar
    ax = axes[1]
    valid_sorted = sorted(valid, key=lambda d: d['series_length'])
    names_s = [d['dataset'] for d in valid_sorted]
    mp_r_s = [d['motif_info']['mp_motif_pair_pearson_r'] for d in valid_sorted]
    c22_r_s = [d['motif_info']['c22_motif_pair_pearson_r'] for d in valid_sorted]

    x = np.arange(len(names_s))
    w = 0.35
    ax.bar(x - w / 2, mp_r_s, w, label='MP Motif Pair r', color=COLOR_MP, alpha=0.85)
    ax.bar(x + w / 2, c22_r_s, w, label='C22 Motif Pair r', color=COLOR_C22, alpha=0.85)
    ax.axhline(y=0, color='k', lw=0.5)
    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_title('Top Motif Pair — Waveform Correlation per Dataset',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names_s, rotation=55, ha='right', fontsize=7)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out, 'slide11b_motif_pair_correlation.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Plot 9: Summary dashboard (slide 12)
# ═════════════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(data, out):
    """Single-page summary with key metrics."""
    summary = data.get('summary', {})
    per_ds = data['per_dataset']
    if not summary:
        print("  SKIP summary_dashboard: no summary data")
        return

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # ── 1: Speedup distribution ──
    c22_su = [d['c22_gpu_speedup'] for d in per_ds if 'c22_gpu_speedup' in d]
    if c22_su:
        ax = fig.add_subplot(gs[0, 0])
        ax.hist(c22_su, bins=max(5, len(c22_su) // 2), color=COLOR_C22, alpha=0.7,
                edgecolor='black')
        ax.axvline(np.mean(c22_su), color='blue', ls='--', lw=2,
                   label=f'Mean: {np.mean(c22_su):.2f}x')
        ax.axvline(np.median(c22_su), color='green', ls='--', lw=2,
                   label=f'Median: {np.median(c22_su):.2f}x')
        ax.axvline(1.0, color='gray', ls=':', lw=1, alpha=0.5)
        ax.set_xlabel('C22 GPU Speedup (x)')
        ax.set_ylabel('Frequency')
        ax.set_title('C22 GPU Speedup Distribution', fontweight='bold')
        ax.legend(fontsize=9)

    # ── 2: Speedup vs series length ──
    if c22_su:
        ax = fig.add_subplot(gs[0, 1])
        lengths = [d['series_length'] for d in per_ds if 'c22_gpu_speedup' in d]
        ax.scatter(lengths, c22_su, color=COLOR_C22, s=50, alpha=0.7, edgecolors='k', lw=0.5)
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel('Series Length (N)')
        ax.set_ylabel('C22 GPU Speedup (x)')
        ax.set_title('C22 GPU Speedup vs Series Length', fontweight='bold')
        ax.set_xscale('log')

    # ── 3: Profile correlation distribution ──
    corrs = [d['profile_correlation'] for d in per_ds if 'profile_correlation' in d]
    if corrs:
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(corrs, bins=max(5, len(corrs) // 2), color='purple', alpha=0.7,
                edgecolor='black')
        ax.axvline(np.mean(corrs), color='blue', ls='--', lw=2,
                   label=f'Mean: {np.mean(corrs):.3f}')
        ax.set_xlabel('Pearson r (C22 vs MP)')
        ax.set_ylabel('Frequency')
        ax.set_title('Profile Correlation Distribution', fontweight='bold')
        ax.legend(fontsize=9)

    # ── 4: C22 vs MP throughput ratio ──
    ratios = []
    for d in per_ds:
        c22_tp = d.get('c22_cpu', {}).get('throughput_mpairs_s')
        mp_tp = d.get('mp_cpu', {}).get('throughput_mpairs_s')
        if c22_tp and mp_tp and c22_tp > 0:
            ratios.append(mp_tp / c22_tp)
    if ratios:
        ax = fig.add_subplot(gs[1, 1])
        names = [d['dataset'] for d in per_ds
                 if d.get('c22_cpu', {}).get('throughput_mpairs_s')
                 and d.get('mp_cpu', {}).get('throughput_mpairs_s')]
        x = np.arange(len(names))
        ax.bar(x, ratios, color='steelblue', alpha=0.8, edgecolor='navy', lw=0.5)
        ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=55, ha='right', fontsize=6)
        ax.set_ylabel('MP / C22 Throughput Ratio')
        ax.set_title('MP is N x faster than C22 (CPU)', fontweight='bold')

    # ── 5: Anomaly detection summary ──
    anom_ds = [d for d in per_ds if d.get('anomaly_detection', {}).get('has_labels')]
    if anom_ds:
        ax = fig.add_subplot(gs[2, 0])
        mp_p = [d['anomaly_detection']['mp_precision_at_k'] for d in anom_ds]
        c22_p = [d['anomaly_detection']['c22_precision_at_k'] for d in anom_ds]
        categories = ['MP Mean', 'C22 Mean', 'MP Median', 'C22 Median']
        values = [np.mean(mp_p), np.mean(c22_p), np.median(mp_p), np.median(c22_p)]
        colors = [COLOR_MP, COLOR_C22, COLOR_MP, COLOR_C22]
        hatches = ['', '', '//', '//']
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_ylabel(f'Precision @ {anom_ds[0]["anomaly_detection"]["top_k"]}')
        ax.set_title(f'Anomaly Detection Summary ({len(anom_ds)} datasets)', fontweight='bold')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f'{val:.0%}', ha='center', fontweight='bold', fontsize=10)

    # ── 6: Key numbers text ──
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    lines = [
        f"Datasets benchmarked: {data['config']['n_datasets']}",
        f"Window size: {data['config']['window_size']}",
        f"CPU threads: {data['config']['cpu_threads']}",
        f"GPU enabled: {data['config']['gpu_enabled']}",
        "",
    ]
    if 'c22_gpu_speedup_mean' in summary:
        lines.append(f"C22 GPU speedup: {summary['c22_gpu_speedup_mean']:.2f}x "
                     f"(range {summary['c22_gpu_speedup_min']:.2f}–"
                     f"{summary['c22_gpu_speedup_max']:.2f})")
    if 'mp_gpu_speedup_mean' in summary:
        lines.append(f"MP  GPU speedup: {summary['mp_gpu_speedup_mean']:.2f}x "
                     f"(range {summary['mp_gpu_speedup_min']:.2f}–"
                     f"{summary['mp_gpu_speedup_max']:.2f})")
    if 'profile_correlation_mean' in summary:
        lines.append(f"Profile correlation: r = {summary['profile_correlation_mean']:.3f} "
                     f"(range {summary['profile_correlation_min']:.3f}–"
                     f"{summary['profile_correlation_max']:.3f})")
    if 'mp_anomaly_precision_mean' in summary:
        lines.append(f"Anomaly precision (MP):  {summary['mp_anomaly_precision_mean']:.0%}")
        lines.append(f"Anomaly precision (C22): {summary['c22_anomaly_precision_mean']:.0%}")
        lines.append(f"  ({summary['n_datasets_with_labels']} labeled datasets)")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('Summary Statistics', fontweight='bold')

    plt.suptitle('20Papers Dataset — C22 vs Matrix Profile Benchmark Summary',
                 fontsize=15, fontweight='bold', y=1.01)
    path = os.path.join(out, 'slide12_summary_dashboard.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plot 20Papers benchmark results from JSON',
    )
    parser.add_argument('--input', '-i', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'benchmark_20papers_full.json'),
                        help='Input JSON from benchmark_20papers_full.py')
    parser.add_argument('--output-dir', '-o', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '20papers_plots'),
                        help='Output directory for plots')
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    data = load_results(args.input)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output:  {out}/\n")

    plot_cpu_gpu_times(data, out)
    plot_throughput_comparison(data, out)
    plot_gpu_speedup(data, out)
    plot_scaling_by_size(data, out)
    plot_thread_scaling(data, out)
    plot_profile_correlation(data, out)
    plot_anomaly_detection(data, out)
    plot_motif_pair_correlation(data, out)
    plot_summary_dashboard(data, out)

    print(f"\nAll plots saved to: {out}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
