# C²² Profile (C22MP) — GPU-Accelerated Feature-Based Time Series Similarity

## Overview

This project extends [SCAMP](https://github.com/zpzim/SCAMP) (SCAlable Matrix Profile) with a new profile type: the **C²² Profile** (C22MP). Instead of comparing subsequences by z-normalized Euclidean distance (as the Matrix Profile does), the C22 Profile represents each subsequence as a **22-dimensional feature vector** using the [catch22](https://github.com/DynamicsAndNeuralSystems/catch22) feature set, then finds the nearest neighbor by **dot-product similarity** in that feature space.

The implementation supports both **multi-threaded CPU** and **CUDA GPU** execution, reusing SCAMP's tiling infrastructure for the GPU dot-product search phase.

### Two-Phase Algorithm

1. **Feature Extraction (O(N))** — For each of the N subsequences, compute all 22 catch22 features. Each feature vector is L2-normalized so dot products yield cosine similarity.
2. **Dot-Product Search (O(N²))** — Find, for every subsequence, the most similar other subsequence (respecting an exclusion zone), producing the C22 Profile and its index.

## Core Source Files

All C22-specific code lives in `src/core/`:

| File | Description |
|---|---|
| `c22_features.h` | `C22FeatureVector` struct (stack-allocated 22-element array) and API for single and batch feature computation. |
| `c22_features.cpp` | CPU implementation of all 22 catch22 features with z-normalization, adaptive parallel extraction. |
| `c22_features_gpu.cu` | CUDA kernels for per-feature GPU computation (includes FFT, splinefit, histogram operations). |
| `c22_profile.h` | Public API: `c22_profile_selfjoin_cpu/gpu` and `c22_profile_abjoin_cpu/gpu`. |
| `c22_profile.cpp` | CPU implementation: multi-threaded feature extraction followed by a tiled, cache-friendly dot-product search with exclusion zone handling. |
| `c22_profile_gpu.cu` | GPU implementation: tiled dot-product kernel using shared memory, packed `uint64` atomics for simultaneous (value, index) updates. |

### Integration Points

| File | Role |
|---|---|
| `src/common/scamp_interface.cpp` | `do_SCAMP_C22()` — dispatches C22 profile computation to CPU or GPU based on the `SCAMPArgs` configuration. |
| `src/python/SCAMP_python.cpp` | Python bindings: exposes `pyscamp.selfjoin_c22()` and `pyscamp.abjoin_c22()`. |

## Python API

```python
import pyscamp

# Self-join (finds nearest neighbor for each subsequence)
profile, index = pyscamp.selfjoin_c22(timeseries, window_size, threads=0, gpu=True)

# AB-join (finds nearest neighbor in B for each subsequence of A)
profile, index = pyscamp.abjoin_c22(a, b, window_size, threads=0, gpu=True)
```

- `threads=0` uses all available CPU cores.
- `gpu=True` runs the dot-product search on GPU (feature extraction always runs on CPU).

## Building

```bash
mkdir build && cd build
cmake .. -DCUDA_ENABLED=1    # omit -DCUDA_ENABLED=1 for CPU-only build
make -j$(nproc)
```

The Python module is built at `build/src/python/pyscamp*.so`. Add it to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/C22/build/src/python:$PYTHONPATH
```

## Test Suite

All C22-specific tests, benchmarks, and demo scripts are in `test/c22/`:

| File | Purpose |
|---|---|
| `test_c22_profile.py` | Core correctness tests (CPU self-join and AB-join). |
| `test_c22_gpu_correctness.py` | GPU vs CPU comparison and brute-force verification. |
| `test_c22_gpu_features.py` | Per-feature GPU vs CPU validation. |
| `test_c22_on_pycatch22_data.py` | Validation against pycatch22's reference outputs. |
| `test_c22_features.cpp` | C++ unit tests for individual feature functions. |
| `run_c22_sample_inputs.py` | Generates ground-truth outputs in `SampleOutputC22/`. |
| `benchmark_20papers_full.py` | Benchmark on the 20Papers datasets (JSON output). |
| `benchmark_sampleinput.py` | Benchmark on SampleInput datasets. |
| `plot_20papers_results.py` | Generates performance charts from benchmark JSON. |
| `plot_benchmark_results.py` | General-purpose plotting from benchmark JSON. |
| `demo1_thread_scaling.py` | Live demo: CPU thread scaling. |
| `demo2_cpu_vs_gpu.py` | Live demo: CPU vs GPU performance comparison. |

### Running Tests

```bash
cd test/c22
export PYTHONPATH=../../build/src/python:$PYTHONPATH

python test_c22_profile.py
python test_c22_gpu_correctness.py    # requires GPU
python test_c22_on_pycatch22_data.py
```

## Benchmark Datasets (20Papers)

The `20Papers /` directory contains 23 time series datasets used in the C²²MP paper (2024). These include ECG, seismic, AIOps, UCR anomaly benchmark, and other real-world datasets ranging from ~100 to ~50,000 data points.

**Download:** [Google Drive — 20Papers datasets](https://drive.google.com/drive/folders/1dw1NUg_5JpBgmtTh8Bzf3oLmrFpa9qSU)

Place the downloaded CSV files in a directory named `20Papers /` at the project root.

### Running Benchmarks

```bash
cd test/c22
export PYTHONPATH=../../build/src/python:$PYTHONPATH

# Run benchmarks (outputs JSON)
python benchmark_20papers_full.py --datasets-dir ../../20Papers\ / --output benchmark_results.json

# Generate charts from results
python plot_20papers_results.py --input benchmark_results.json --output-dir ./charts/
```
