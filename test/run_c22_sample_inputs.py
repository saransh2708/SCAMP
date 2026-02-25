#!/usr/bin/env python3
"""
Run C22 Profile on SampleInput files and write results to SampleOutputC22/.

Output format (one line per subsequence):
  <profile_value> <best_match_index>

Usage:
  # CPU only:
  PYTHONPATH=build/src/python python3 test/run_c22_sample_inputs.py

  # With GPU (on cerebro):
  PYTHONPATH=build/src/python python3 test/run_c22_sample_inputs.py --gpu
"""

import sys
import os
import time
import argparse

try:
    import pyscamp
except ImportError:
    print("ERROR: pyscamp not found.")
    print("Run from repo root: PYTHONPATH=build/src/python python3 test/run_c22_sample_inputs.py")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true",
                    help="Use GPU if available (default: CPU only)")
parser.add_argument("--window", "-m", type=int, default=100,
                    help="Subsequence window size (default: 100)")
parser.add_argument("--max-n", type=int, default=65536,
                    help="Skip inputs larger than this (default: 65536). "
                         "Use 0 for no limit.")
args = parser.parse_args()

use_gpu = args.gpu and pyscamp.gpu_supported()
window = args.window
max_n = args.max_n  # 0 means unlimited

print(f"GPU support compiled in: {pyscamp.gpu_supported()}")
print(f"Using GPU: {use_gpu}")
print(f"Window size: {window}")
print(f"Max series length: {'unlimited' if max_n == 0 else max_n}")
print()

# ---------------------------------------------------------------------------
# Input / output directories
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir  = os.path.join(script_dir, "SampleInput")
output_dir = os.path.join(script_dir, "SampleOutputC22")
os.makedirs(output_dir, exist_ok=True)

# Select a representative subset — skip very large files unless --max-n 0
input_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))

# ---------------------------------------------------------------------------
# Helper: load a time series file (one float per line, skip NaN lines)
# ---------------------------------------------------------------------------
def load_ts(path):
    values = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = float(line)
                values.append(v)
            except ValueError:
                pass  # skip non-numeric lines (e.g. headers)
    return values

# ---------------------------------------------------------------------------
# Helper: write output
# ---------------------------------------------------------------------------
def write_output(path, profile, index):
    with open(path, "w") as f:
        f.write(f"# C22 Profile  window={window}  gpu={use_gpu}\n")
        f.write(f"# profile_value  best_match_index\n")
        for p, i in zip(profile, index):
            f.write(f"{p:.6f}\t{i}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"{'File':<40} {'N':>8} {'Subs':>8} {'Time':>8}  {'Output'}")
print("-" * 80)

total_pass = 0
total_skip = 0
total_fail = 0

for fname in input_files:
    in_path = os.path.join(input_dir, fname)
    out_name = fname.replace(".txt", f"_c22_w{window}.txt")
    out_path = os.path.join(output_dir, out_name)

    # Load
    ts = load_ts(in_path)
    N = len(ts)
    n_subs = N - window + 1

    # Skip if too large
    if max_n > 0 and N > max_n:
        print(f"  {'[SKIP]':<6} {fname:<40} {N:>8}  (exceeds --max-n {max_n})")
        total_skip += 1
        continue

    # Skip if too short
    if n_subs < 2:
        print(f"  {'[SKIP]':<6} {fname:<40} {N:>8}  (too short for window={window})")
        total_skip += 1
        continue

    # Compute
    try:
        t0 = time.time()
        profile, index = pyscamp.selfjoin_c22(ts, window, gpu=use_gpu)
        elapsed = time.time() - t0

        write_output(out_path, profile, index)

        print(f"  [OK]   {fname:<40} {N:>8} {n_subs:>8} {elapsed:>7.2f}s  -> {out_name}")
        total_pass += 1

    except Exception as e:
        print(f"  [FAIL] {fname:<40} {N:>8}  ERROR: {e}")
        total_fail += 1

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print(f"  Processed: {total_pass}  Skipped: {total_skip}  Failed: {total_fail}")
print(f"  Output directory: {output_dir}/")
print("=" * 80)

if total_fail > 0:
    sys.exit(1)
