#!/usr/bin/env python3
"""
Benchmark C²²MP on datasets from the 2024 paper.
Measures CPU and GPU performance and outputs JSON results for plotting.

IMPORTANT: This script ONLY outputs JSON - no plots are generated.
Use plot_benchmark_results.py on your local machine to generate charts.

Usage:
    PYTHONPATH=build/src/python python3 test/benchmark_c22_datasets.py [--datasets-dir path] [--window-size m] [--gpu] [--cpu-only]
    
Output:
    - test/benchmark_results_datasets.json (JSON with all results - no plots)
"""

import sys
import os
import json
import time
import argparse
import multiprocessing
import csv
from pathlib import Path

import numpy as np

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Add the build directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found. Build first, then run with PYTHONPATH=build/src/python")
    sys.exit(1)

# Configuration
REPEAT = 3  # Number of repetitions per measurement (take min)
DEFAULT_WINDOW = 100  # Default window size if not specified
MIN_SERIES_LENGTH = 200  # Skip datasets shorter than this

def load_timeseries(csv_path):
    """Load time series from CSV file. Assumes 'ts' column or first numeric column."""
    try:
        if HAS_PANDAS:
            # Use pandas if available (more robust)
            df = pd.read_csv(csv_path)
            
            # Try to find 'ts' column first
            if 'ts' in df.columns:
                ts = df['ts'].values
            else:
                # Find first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ts = df[numeric_cols[0]].values
                else:
                    raise ValueError(f"No numeric column found in {csv_path}")
            
            # Remove NaN values and convert to list
            ts = ts[~np.isnan(ts)].tolist()
        else:
            # Fallback to csv module (standard library)
            ts = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                # Try to find 'ts' column first
                if 'ts' in reader.fieldnames:
                    col_name = 'ts'
                else:
                    # Use first column
                    col_name = reader.fieldnames[0] if reader.fieldnames else None
                    if col_name is None:
                        raise ValueError(f"No columns found in {csv_path}")
                
                for row in reader:
                    try:
                        val = float(row[col_name])
                        if not np.isnan(val):
                            ts.append(val)
                    except (ValueError, KeyError):
                        continue  # Skip non-numeric or missing values
        
        if len(ts) < MIN_SERIES_LENGTH:
            return None, f"Series too short ({len(ts)} < {MIN_SERIES_LENGTH})"
        
        return ts, None
    except Exception as e:
        return None, str(e)


def bench(func, *args, repeat=REPEAT, **kwargs):
    """Run func(*args, **kwargs) `repeat` times; return min elapsed time."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            func(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"    ERROR during benchmark: {e}", file=sys.stderr)
            return None
    return min(times) if times else None


def compute_throughput(n_subseq, time_s):
    """Compute throughput in million pairs per second."""
    if time_s is None or time_s <= 0:
        return None
    pairs = n_subseq * n_subseq
    return (pairs / time_s) / 1e6  # Convert to millions


def benchmark_dataset(dataset_path, window_size, use_cpu=True, use_gpu=False, num_threads=None):
    """Benchmark C²²MP on a single dataset."""
    dataset_name = Path(dataset_path).stem
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Load time series
    ts, error = load_timeseries(dataset_path)
    if ts is None:
        print(f"  SKIPPED: {error}")
        return None
    
    n = len(ts)
    if n <= window_size:
        print(f"  SKIPPED: Series length ({n}) <= window size ({window_size})")
        return None
    
    n_subseq = n - window_size + 1
    
    print(f"  Series length: {n}")
    print(f"  Window size: {window_size}")
    print(f"  Subsequences: {n_subseq}")
    
    result = {
        'dataset': dataset_name,
        'dataset_path': str(dataset_path),
        'series_length': n,
        'window_size': window_size,
        'n_subsequences': n_subseq,
        'cpu': {},
        'gpu': {}
    }
    
    # CPU benchmark
    if use_cpu:
        print(f"  Running CPU benchmark ({num_threads or 'auto'} threads)...", end=" ", flush=True)
        t_cpu = bench(pyscamp.selfjoin_c22, ts, window_size, threads=num_threads, gpu=False)
        if t_cpu is not None:
            throughput_cpu = compute_throughput(n_subseq, t_cpu)
            result['cpu'] = {
                'time_s': t_cpu,
                'throughput_mpairs_s': throughput_cpu,
                'threads': num_threads or multiprocessing.cpu_count()
            }
            print(f"✓ {t_cpu:.3f}s ({throughput_cpu:.2f} M pairs/s)")
        else:
            print("✗ FAILED")
            result['cpu'] = {'error': 'Benchmark failed'}
    
    # GPU benchmark
    if use_gpu and pyscamp.gpu_supported():
        print(f"  Running GPU benchmark...", end=" ", flush=True)
        t_gpu = bench(pyscamp.selfjoin_c22, ts, window_size, threads=num_threads, gpu=True)
        if t_gpu is not None:
            throughput_gpu = compute_throughput(n_subseq, t_gpu)
            result['gpu'] = {
                'time_s': t_gpu,
                'throughput_mpairs_s': throughput_gpu,
                'threads': num_threads or multiprocessing.cpu_count()
            }
            print(f"✓ {t_gpu:.3f}s ({throughput_gpu:.2f} M pairs/s)")
            
            # Compute speedup if CPU result is available
            if use_cpu and 'time_s' in result['cpu']:
                speedup = result['cpu']['time_s'] / t_gpu
                result['speedup'] = speedup
                print(f"  GPU speedup: {speedup:.2f}×")
        else:
            print("✗ FAILED")
            result['gpu'] = {'error': 'Benchmark failed'}
    elif use_gpu:
        print("  GPU not available (skipped)")
        result['gpu'] = {'error': 'GPU not supported'}
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark C²²MP on datasets from the 2024 paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all datasets with default settings (CPU only)
  PYTHONPATH=build/src/python python3 test/benchmark_c22_datasets.py
  
  # Run with GPU
  PYTHONPATH=build/src/python python3 test/benchmark_c22_datasets.py --gpu
  
  # Run on specific directory with custom window size
  PYTHONPATH=build/src/python python3 test/benchmark_c22_datasets.py \\
      --datasets-dir "20Papers /" --window-size 50
        """
    )
    
    parser.add_argument('--datasets-dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), '..', '20Papers '),
                       help='Directory containing CSV datasets (default: ../20Papers /)')
    parser.add_argument('--window-size', '-m', type=int, default=DEFAULT_WINDOW,
                       help=f'Window size for C²²MP (default: {DEFAULT_WINDOW})')
    parser.add_argument('--gpu', action='store_true',
                       help='Also benchmark GPU (default: CPU only)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU only (overrides --gpu)')
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of CPU threads (default: auto)')
    parser.add_argument('--output', '-o', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'benchmark_results_datasets.json'),
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Determine compute devices
    use_cpu = True
    use_gpu = args.gpu and not args.cpu_only and pyscamp.gpu_supported()
    
    # Print configuration
    print("="*70)
    print("C²²MP Dataset Benchmarking")
    print("="*70)
    print(f"Datasets directory: {args.datasets_dir}")
    print(f"Window size: {args.window_size}")
    print(f"CPU: {'✓' if use_cpu else '✗'}")
    print(f"GPU: {'✓' if use_gpu else '✗'}")
    if use_cpu:
        print(f"CPU threads: {args.threads or 'auto'}")
    print(f"Output file: {args.output}")
    print("="*70)
    
    # Find all CSV files
    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        print(f"ERROR: Datasets directory not found: {datasets_dir}", file=sys.stderr)
        sys.exit(1)
    
    csv_files = sorted(datasets_dir.glob('*.csv'))
    if not csv_files:
        print(f"ERROR: No CSV files found in {datasets_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    # Benchmark each dataset
    all_results = {
        'config': {
            'window_size': args.window_size,
            'cpu_enabled': use_cpu,
            'gpu_enabled': use_gpu,
            'cpu_threads': args.threads or multiprocessing.cpu_count(),
            'repeat': REPEAT,
            'min_series_length': MIN_SERIES_LENGTH
        },
        'datasets': []
    }
    
    successful = 0
    skipped = 0
    failed = 0
    
    for csv_file in csv_files:
        result = benchmark_dataset(csv_file, args.window_size, use_cpu, use_gpu, args.threads)
        if result is None:
            skipped += 1
        elif 'error' in result.get('cpu', {}) and 'error' in result.get('gpu', {}):
            failed += 1
        else:
            all_results['datasets'].append(result)
            successful += 1
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"Total datasets: {len(csv_files)}")
    print(f"  Successful: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"\nResults saved to: {output_path}")
    
    # Print speedup summary if GPU was used
    if use_gpu:
        speedups = [r.get('speedup') for r in all_results['datasets'] if 'speedup' in r]
        if speedups:
            print(f"\nGPU Speedup Summary:")
            print(f"  Mean: {np.mean(speedups):.2f}×")
            print(f"  Median: {np.median(speedups):.2f}×")
            print(f"  Min: {np.min(speedups):.2f}×")
            print(f"  Max: {np.max(speedups):.2f}×")
    
    print("="*70)


if __name__ == '__main__':
    main()
