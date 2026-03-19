#include "c22_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "c22_features.h"

namespace SCAMP {

// ============================================================================
// Internal: Flatten C22FeatureVectors into a contiguous double array (N x 22)
// for cache-friendly dot product computation.
// ============================================================================
static std::vector<double> flatten_features(
    const std::vector<C22FeatureVector>& vecs) {
  const int D = C22FeatureVector::NUM_FEATURES;
  std::vector<double> flat(vecs.size() * D);
  for (size_t i = 0; i < vecs.size(); ++i) {
    for (int f = 0; f < D; ++f) {
      flat[i * D + f] = vecs[i].features[f];
    }
  }
  return flat;
}

// ============================================================================
// Internal: Compute dot product between two 22-element feature vectors.
// Kept simple for auto-vectorization by the compiler.
// ============================================================================
static inline double dot22(const double* __restrict__ a,
                           const double* __restrict__ b) {
  double sum = 0.0;
  for (int f = 0; f < C22FeatureVector::NUM_FEATURES; ++f) {
    sum += a[f] * b[f];
  }
  return sum;
}

// ============================================================================
// CPU Self-Join
// ============================================================================
C22ProfileResult c22_profile_selfjoin_cpu(const std::vector<double>& timeseries,
                                          int window_size, int num_threads) {
  const int N = static_cast<int>(timeseries.size()) - window_size + 1;
  if (N <= 0) {
    return C22ProfileResult();
  }

  const int exclusion = window_size / 4;

  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 1;
  }

  std::vector<C22FeatureVector> features =
      compute_c22_vectors_parallel(timeseries, window_size, num_threads);

  const int D = C22FeatureVector::NUM_FEATURES;
  std::vector<double> F = flatten_features(features);

  std::vector<double> profile(N, -std::numeric_limits<double>::infinity());
  std::vector<int> index(N, -1);

  std::vector<std::thread> threads;
  int rows_per_thread = (N + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int row_start = t * rows_per_thread;
    int row_end = std::min(row_start + rows_per_thread, N);
    if (row_start >= N) break;

    threads.emplace_back(
        [&F, &profile, &index, N, D, exclusion, row_start, row_end]() {
          for (int i = row_start; i < row_end; ++i) {
            const double* fi = &F[i * D];
            double best_dot = -std::numeric_limits<double>::infinity();
            int best_j = -1;

            for (int j = 0; j < N; ++j) {
              int diff = (i > j) ? (i - j) : (j - i);
              if (diff <= exclusion) continue;

              double d = dot22(fi, &F[j * D]);
              if (d > best_dot) {
                best_dot = d;
                best_j = j;
              }
            }

            profile[i] = best_dot;
            index[i] = best_j;
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  C22ProfileResult result;
  result.profile = std::move(profile);
  result.index = std::move(index);
  return result;
}

// ============================================================================
// CPU AB-Join
// ============================================================================
C22ProfileResult c22_profile_abjoin_cpu(const std::vector<double>& timeseries_a,
                                        const std::vector<double>& timeseries_b,
                                        int window_size, int num_threads) {
  const int NA = static_cast<int>(timeseries_a.size()) - window_size + 1;
  const int NB = static_cast<int>(timeseries_b.size()) - window_size + 1;
  if (NA <= 0 || NB <= 0) {
    return C22ProfileResult();
  }

  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 1;
  }

  std::vector<C22FeatureVector> features_a =
      compute_c22_vectors_parallel(timeseries_a, window_size, num_threads);
  std::vector<C22FeatureVector> features_b =
      compute_c22_vectors_parallel(timeseries_b, window_size, num_threads);

  const int D = C22FeatureVector::NUM_FEATURES;
  std::vector<double> FA = flatten_features(features_a);
  std::vector<double> FB = flatten_features(features_b);

  std::vector<double> profile(NA, -std::numeric_limits<double>::infinity());
  std::vector<int> index(NA, -1);

  std::vector<std::thread> threads;
  int rows_per_thread = (NA + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int row_start = t * rows_per_thread;
    int row_end = std::min(row_start + rows_per_thread, NA);
    if (row_start >= NA) break;

    threads.emplace_back(
        [&FA, &FB, &profile, &index, D, NB, row_start, row_end]() {
          for (int i = row_start; i < row_end; ++i) {
            const double* fi = &FA[i * D];
            double best_dot = -std::numeric_limits<double>::infinity();
            int best_j = -1;

            for (int j = 0; j < NB; ++j) {
              double d = dot22(fi, &FB[j * D]);
              if (d > best_dot) {
                best_dot = d;
                best_j = j;
              }
            }

            profile[i] = best_dot;
            index[i] = best_j;
          }
        });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  C22ProfileResult result;
  result.profile = std::move(profile);
  result.index = std::move(index);
  return result;
}

// ============================================================================
// GPU Implementation — delegates to CUDA kernels defined in c22_profile_gpu.cu
//
// Resource-aware strategy:
//   window ≤ 256: ALL-GPU path — features AND dot product on GPU.
//                 The raw timeseries is transferred to the GPU; the GPU
//                 computes N×22 features (one thread per subsequence), then
//                 runs the tiled dot product search, all without any CPU
//                 feature work.  PCIe traffic is minimised (N*8 bytes in vs
//                 N*22*8 bytes if features were computed on CPU first).
//   window > 256: HYBRID path — features on CPU (multi-threaded), dot product
//                 on GPU.  Falls back gracefully because the GPU kernel uses
//                 fixed stack arrays sized for window ≤ 256.
// ============================================================================

#ifdef _HAS_CUDA_

// The maximum window size supported by the GPU feature kernel (must match
// C22_GPU_MAX_W in c22_features_gpu.cu).
// GPU feature extraction path (c22_features_gpu.cu) is validated for window
// sizes up to 256.  Windows above this fall back to the hybrid path (CPU
// features + GPU dot product) because the GPU kernel uses fixed stack arrays
// sized for W ≤ 256.
static constexpr int kGpuMaxWindow = 256;

extern "C" {
// All-GPU paths (raw timeseries → features on GPU → dot product on GPU)
void c22_profile_selfjoin_gpu_launch_from_ts(const double* h_ts, int ts_length,
                                             int window, double* h_profile,
                                             int* h_index, int N, int exclusion,
                                             int gpu_id);
void c22_profile_abjoin_gpu_launch_from_ts(const double* h_ts_a,
                                           int ts_a_length,
                                           const double* h_ts_b,
                                           int ts_b_length, int window,
                                           double* h_profile, int* h_index,
                                           int NA, int NB, int gpu_id);
// Hybrid paths (pre-computed CPU features → dot product on GPU)
void c22_profile_selfjoin_gpu_launch(const double* h_features,
                                     double* h_profile, int* h_index, int N,
                                     int exclusion, int gpu_id);
void c22_profile_abjoin_gpu_launch(const double* h_features_a,
                                   const double* h_features_b,
                                   double* h_profile, int* h_index, int NA,
                                   int NB, int gpu_id);
}

C22ProfileResult c22_profile_selfjoin_gpu(const std::vector<double>& timeseries,
                                          int window_size, int gpu_id,
                                          int num_cpu_threads) {
  const int N = static_cast<int>(timeseries.size()) - window_size + 1;
  if (N <= 0) return C22ProfileResult();

  const int exclusion = window_size / 4;
  std::vector<double> profile(N);
  std::vector<int> index(N);

  if (window_size <= kGpuMaxWindow) {
    // ── All-GPU path: raw TS → GPU features + GPU dot product ────────────
    c22_profile_selfjoin_gpu_launch_from_ts(
        timeseries.data(), static_cast<int>(timeseries.size()), window_size,
        profile.data(), index.data(), N, exclusion, gpu_id);
  } else {
    // ── Hybrid path: CPU features → GPU dot product ───────────────────────
    if (num_cpu_threads <= 0) {
      num_cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (num_cpu_threads <= 0) num_cpu_threads = 1;
    }
    std::vector<C22FeatureVector> features =
        compute_c22_vectors_parallel(timeseries, window_size, num_cpu_threads);
    std::vector<double> F = flatten_features(features);
    c22_profile_selfjoin_gpu_launch(F.data(), profile.data(), index.data(), N,
                                    exclusion, gpu_id);
  }

  C22ProfileResult result;
  result.profile = std::move(profile);
  result.index = std::move(index);
  return result;
}

C22ProfileResult c22_profile_abjoin_gpu(const std::vector<double>& timeseries_a,
                                        const std::vector<double>& timeseries_b,
                                        int window_size, int gpu_id,
                                        int num_cpu_threads) {
  const int NA = static_cast<int>(timeseries_a.size()) - window_size + 1;
  const int NB = static_cast<int>(timeseries_b.size()) - window_size + 1;
  if (NA <= 0 || NB <= 0) return C22ProfileResult();

  std::vector<double> profile(NA);
  std::vector<int> index(NA);

  if (window_size <= kGpuMaxWindow) {
    // ── All-GPU path ───────────────────────────────────────────────────────
    c22_profile_abjoin_gpu_launch_from_ts(
        timeseries_a.data(), static_cast<int>(timeseries_a.size()),
        timeseries_b.data(), static_cast<int>(timeseries_b.size()), window_size,
        profile.data(), index.data(), NA, NB, gpu_id);
  } else {
    // ── Hybrid path ────────────────────────────────────────────────────────
    if (num_cpu_threads <= 0) {
      num_cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (num_cpu_threads <= 0) num_cpu_threads = 1;
    }
    std::vector<C22FeatureVector> features_a = compute_c22_vectors_parallel(
        timeseries_a, window_size, num_cpu_threads);
    std::vector<C22FeatureVector> features_b = compute_c22_vectors_parallel(
        timeseries_b, window_size, num_cpu_threads);
    std::vector<double> FA = flatten_features(features_a);
    std::vector<double> FB = flatten_features(features_b);
    c22_profile_abjoin_gpu_launch(FA.data(), FB.data(), profile.data(),
                                  index.data(), NA, NB, gpu_id);
  }

  C22ProfileResult result;
  result.profile = std::move(profile);
  result.index = std::move(index);
  return result;
}

#endif  // _HAS_CUDA_

}  // namespace SCAMP
