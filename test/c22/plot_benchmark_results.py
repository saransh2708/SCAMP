#!/usr/bin/env python3
"""
Plot benchmarking results from JSON file.
Generates performance comparison charts for C²²MP on datasets.

NOTE: This script is intended to be run on your LOCAL machine after pulling
the JSON results from the remote GPU host via git. It requires matplotlib.

Usage:
    python3 test/c22/plot_benchmark_results.py [--input benchmark_results_datasets.json] [--output-dir plots/]
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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("ERROR: matplotlib not found. Install with: pip install matplotlib")
    sys.exit(1)


def load_results(json_path):
    """Load benchmark results from JSON file or stdin."""
    if json_path == '-' or json_path == '/dev/stdin':
        # Read from stdin
        return json.load(sys.stdin)
    else:
        # Read from file
        with open(json_path, 'r') as f:
            return json.load(f)


def plot_performance_comparison(results, output_dir):
    """Plot CPU vs GPU performance comparison."""
    datasets = results['datasets']
    
    # Filter datasets with both CPU and GPU results
    valid = [d for d in datasets if 'time_s' in d.get('cpu', {}) and 'time_s' in d.get('gpu', {})]
    
    if not valid:
        print("WARNING: No datasets with both CPU and GPU results found.")
        return
    
    dataset_names = [d['dataset'] for d in valid]
    cpu_times = [d['cpu']['time_s'] for d in valid]
    gpu_times = [d['gpu']['time_s'] for d in valid]
    speedups = [d.get('speedup', 1.0) for d in valid]
    
    # Sort by CPU time (descending)
    sorted_indices = np.argsort(cpu_times)[::-1]
    dataset_names = [dataset_names[i] for i in sorted_indices]
    cpu_times = [cpu_times[i] for i in sorted_indices]
    gpu_times = [gpu_times[i] for i in sorted_indices]
    speedups = [speedups[i] for i in sorted_indices]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time comparison (log scale)
    ax1 = axes[0]
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='darkorange', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('C²²MP Performance: CPU vs GPU', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=9)
    ax1.set_yscale('log')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s',
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    # Plot 2: Speedup
    ax2 = axes[1]
    colors = ['green' if s >= 1.0 else 'red' for s in speedups]
    bars = ax2.bar(x, speedups, color=colors, alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Dataset', fontsize=12)
    ax2.set_ylabel('GPU Speedup (×)', fontsize=12)
    ax2.set_title('GPU Speedup Over CPU', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved {output_path}")


def plot_throughput_comparison(results, output_dir):
    """Plot throughput comparison (M pairs/s)."""
    datasets = results['datasets']
    
    # Filter datasets with both CPU and GPU results
    valid = [d for d in datasets if 'throughput_mpairs_s' in d.get('cpu', {}) 
             and 'throughput_mpairs_s' in d.get('gpu', {})]
    
    if not valid:
        print("WARNING: No datasets with both CPU and GPU throughput results found.")
        return
    
    dataset_names = [d['dataset'] for d in valid]
    cpu_throughput = [d['cpu']['throughput_mpairs_s'] for d in valid]
    gpu_throughput = [d['gpu']['throughput_mpairs_s'] for d in valid]
    
    # Sort by CPU throughput (ascending)
    sorted_indices = np.argsort(cpu_throughput)
    dataset_names = [dataset_names[i] for i in sorted_indices]
    cpu_throughput = [cpu_throughput[i] for i in sorted_indices]
    gpu_throughput = [gpu_throughput[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpu_throughput, width, label='CPU', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, gpu_throughput, width, label='GPU', color='darkorange', alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Throughput (M pairs/s)', fontsize=12)
    ax.set_title('C²²MP Throughput: CPU vs GPU', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'throughput_comparison.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved {output_path}")


def plot_scaling_by_size(results, output_dir):
    """Plot performance vs series length."""
    datasets = results['datasets']
    
    # CPU results
    cpu_valid = [(d['series_length'], d['cpu']['time_s']) 
                 for d in datasets if 'time_s' in d.get('cpu', {})]
    gpu_valid = [(d['series_length'], d['gpu']['time_s']) 
                 for d in datasets if 'time_s' in d.get('gpu', {})]
    
    if not cpu_valid and not gpu_valid:
        print("WARNING: No valid timing results found.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if cpu_valid:
        cpu_lengths, cpu_times = zip(*sorted(cpu_valid))
        ax.scatter(cpu_lengths, cpu_times, label='CPU', color='steelblue', s=60, alpha=0.7)
        # Fit a trend line (log-log)
        if len(cpu_lengths) > 1:
            log_lengths = np.log10(cpu_lengths)
            log_times = np.log10(cpu_times)
            z = np.polyfit(log_lengths, log_times, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(min(cpu_lengths)), np.log10(max(cpu_lengths)), 100)
            y_trend = 10**p(np.log10(x_trend))
            ax.plot(x_trend, y_trend, '--', color='steelblue', alpha=0.5, linewidth=1)
    
    if gpu_valid:
        gpu_lengths, gpu_times = zip(*sorted(gpu_valid))
        ax.scatter(gpu_lengths, gpu_times, label='GPU', color='darkorange', s=60, alpha=0.7)
        # Fit a trend line
        if len(gpu_lengths) > 1:
            log_lengths = np.log10(gpu_lengths)
            log_times = np.log10(gpu_times)
            z = np.polyfit(log_lengths, log_times, 1)
            p = np.poly1d(z)
            x_trend = np.logspace(np.log10(min(gpu_lengths)), np.log10(max(gpu_lengths)), 100)
            y_trend = 10**p(np.log10(x_trend))
            ax.plot(x_trend, y_trend, '--', color='darkorange', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Series Length (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('C²²MP Performance vs Series Length', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scaling_by_size.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved {output_path}")


def plot_summary_statistics(results, output_dir):
    """Plot summary statistics."""
    datasets = results['datasets']
    
    # Collect statistics
    cpu_times = [d['cpu']['time_s'] for d in datasets if 'time_s' in d.get('cpu', {})]
    gpu_times = [d['gpu']['time_s'] for d in datasets if 'time_s' in d.get('gpu', {})]
    speedups = [d.get('speedup') for d in datasets if 'speedup' in d]
    
    if not speedups:
        print("WARNING: No speedup data available for summary.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Speedup distribution
    ax1 = axes[0]
    ax1.hist(speedups, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(speedups), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speedups):.2f}×')
    ax1.axvline(np.median(speedups), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(speedups):.2f}×')
    ax1.set_xlabel('GPU Speedup (×)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('GPU Speedup Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Speedup vs Series Length
    ax2 = axes[1]
    if speedups:
        lengths = [d['series_length'] for d in datasets if 'speedup' in d]
        ax2.scatter(lengths, speedups, color='darkorange', s=60, alpha=0.7)
        ax2.set_xlabel('Series Length (N)', fontsize=12)
        ax2.set_ylabel('GPU Speedup (×)', fontsize=12)
        ax2.set_title('Speedup vs Series Length', fontsize=13, fontweight='bold')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'summary_statistics.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot C²²MP benchmarking results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'benchmark_results_datasets.json'),
                       help='Input JSON file with benchmark results')
    parser.add_argument('--output-dir', '-o', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'benchmark_plots'),
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    try:
        results = load_results(args.input)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in: {output_dir}")
    print()
    
    # Generate all plots
    plot_performance_comparison(results, output_dir)
    plot_throughput_comparison(results, output_dir)
    plot_scaling_by_size(results, output_dir)
    plot_summary_statistics(results, output_dir)
    
    print()
    print("="*70)
    print("PLOTTING COMPLETE")
    print(f"Plots saved to: {output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
