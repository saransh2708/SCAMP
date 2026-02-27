#!/usr/bin/env python3
"""
run_c22_gcp.py — GCP entry-point for C22 Profile computation.

Usage
-----
  python3 run_c22_gcp.py --input <file> --window <W> [options]

Arguments
---------
  --input   PATH    Input time series (CSV or whitespace-separated values,
                    one number per line or comma-separated on a single line).
  --output  PATH    Output path for profile results (CSV). Defaults to stdout.
  --window  INT     Subsequence window size (required).
  --threads INT     Number of CPU threads (0 = auto-detect, default 0).
  --gpu     INT     GPU device ID to use (-1 = CPU-only, default -1).
  --abjoin  PATH    Optional second time series for AB-join. When omitted,
                    a self-join is performed.

Output format (CSV)
-------------------
  index,profile_value,nn_index
  0,<dot_product>,<best_match_index>
  1,...
  ...

Exit codes
----------
  0   Success
  1   Input error (bad file, bad window, etc.)
  2   Computation error
"""

import argparse
import os
import sys
import time


def _add_pyscamp_to_path():
    """Add the pyscamp build directory to sys.path."""
    candidates = [
        # Running inside Docker image
        "/app/build/src/python",
        # Running from repo root after local build
        os.path.join(os.path.dirname(__file__), "..", "build", "src", "python"),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _load_timeseries(path):
    """
    Load a time series from *path*.

    Accepts:
    - One float per line (plain text / CSV with a single column).
    - Multiple floats per line separated by commas or whitespace
      (only the first value on each line is used if multiple columns exist).
    """
    values = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support comma-separated or whitespace-separated
            parts = line.replace(",", " ").split()
            try:
                values.append(float(parts[0]))
            except (ValueError, IndexError) as exc:
                print(
                    f"ERROR: cannot parse float on line {lineno} of '{path}': {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
    return values


def _write_output(profile, index, output_path):
    """Write profile results to *output_path* (or stdout if None)."""
    lines = ["index,profile_value,nn_index"]
    for i, (pv, ni) in enumerate(zip(profile, index)):
        lines.append(f"{i},{pv:.10g},{ni}")
    content = "\n".join(lines) + "\n"

    if output_path is None:
        sys.stdout.write(content)
    else:
        with open(output_path, "w") as fh:
            fh.write(content)
        print(f"Results written to: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Compute C22 Profile (catch22 Matrix Profile) on GCP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="Input time series file.")
    parser.add_argument("--abjoin", default=None, metavar="PATH",
                        help="Second time series for AB-join (omit for self-join).")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Output CSV file (default: stdout).")
    parser.add_argument("--window", type=int, required=True, metavar="W",
                        help="Subsequence window size.")
    parser.add_argument("--threads", type=int, default=0, metavar="N",
                        help="CPU thread count (0 = auto-detect).")
    parser.add_argument("--gpu", type=int, default=-1, metavar="ID",
                        help="GPU device ID (-1 = CPU-only).")
    args = parser.parse_args()

    # Validate arguments
    if args.window < 4:
        print("ERROR: --window must be at least 4.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.input):
        print(f"ERROR: input file not found: '{args.input}'", file=sys.stderr)
        sys.exit(1)

    if args.abjoin is not None and not os.path.isfile(args.abjoin):
        print(f"ERROR: abjoin file not found: '{args.abjoin}'", file=sys.stderr)
        sys.exit(1)

    # Load pyscamp
    _add_pyscamp_to_path()
    try:
        import pyscamp  # noqa: PLC0415
    except ImportError as exc:
        print(
            f"ERROR: could not import pyscamp — is the build directory on "
            f"PYTHONPATH?\n  {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Load time series
    ts = _load_timeseries(args.input)
    if len(ts) < args.window:
        print(
            f"ERROR: time series length ({len(ts)}) is shorter than "
            f"--window ({args.window}).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"Loaded time series: {len(ts)} points from '{args.input}'",
        file=sys.stderr,
    )

    ts_b = None
    if args.abjoin is not None:
        ts_b = _load_timeseries(args.abjoin)
        if len(ts_b) < args.window:
            print(
                f"ERROR: abjoin time series length ({len(ts_b)}) is shorter "
                f"than --window ({args.window}).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Loaded AB-join time series: {len(ts_b)} points from '{args.abjoin}'",
            file=sys.stderr,
        )

    # Determine compute path
    use_gpu = args.gpu >= 0
    join_type = "AB-join" if ts_b is not None else "self-join"
    compute_device = f"GPU (device {args.gpu})" if use_gpu else "CPU"
    print(
        f"Running C22 {join_type} | window={args.window} | "
        f"threads={args.threads or 'auto'} | device={compute_device}",
        file=sys.stderr,
    )

    t0 = time.perf_counter()
    try:
        kwargs = {"threads": args.threads}
        if use_gpu:
            kwargs["gpu_id"] = args.gpu

        if ts_b is not None:
            profile, index = pyscamp.abjoin_c22(ts, ts_b, args.window, **kwargs)
        else:
            profile, index = pyscamp.selfjoin_c22(ts, args.window, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: computation failed: {exc}", file=sys.stderr)
        sys.exit(2)

    elapsed = time.perf_counter() - t0
    n_subseq = len(ts) - args.window + 1
    print(
        f"Done in {elapsed:.2f}s | {n_subseq} subsequences",
        file=sys.stderr,
    )

    _write_output(profile, index, args.output)


if __name__ == "__main__":
    main()
