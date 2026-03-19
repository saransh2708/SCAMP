#include "c22_features.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

// pycatch22 C functions (CPU-compatible, no FFTW dependency)
// Signatures must exactly match those in pycatch22/src/C/*.c
extern "C" {
// DN features
extern double DN_HistogramMode_5(const double y[], const int size);
extern double DN_HistogramMode_10(const double y[], const int size);
extern double DN_OutlierInclude_p_001_mdrmd(const double y[], const int size);
extern double DN_OutlierInclude_n_001_mdrmd(const double y[], const int size);

// CO features
extern double CO_f1ecac(const double y[], const int size);
extern int CO_FirstMin_ac(const double y[], const int size);
extern double CO_HistogramAMI_even_2_5(const double y[], const int size);
extern double CO_trev_1_num(const double y[], const int size);
extern double CO_Embed2_Dist_tau_d_expfit_meandiff(const double y[],
                                                   const int size);

// MD features
extern double MD_hrv_classic_pnn40(const double y[], const int size);

// SB features
extern double SB_BinaryStats_mean_longstretch1(const double y[],
                                               const int size);
extern double SB_BinaryStats_diff_longstretch0(const double y[],
                                               const int size);
extern double SB_TransitionMatrix_3ac_sumdiagcov(const double y[],
                                                 const int size);
extern double SB_MotifThree_quantile_hh(const double y[], const int size);

// PD features — NOTE: returns int, not double
extern int PD_PeriodicityWang_th0_01(const double y[], const int size);

// IN features
extern double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double y[],
                                                      const int size);

// FC features
extern double FC_LocalSimple_mean1_tauresrat(const double y[], const int size);
extern double FC_LocalSimple_mean3_stderr(const double y[], const int size);

// SC features
extern double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double y[],
                                                          const int size);
extern double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double y[],
                                                     const int size);

// SP features
extern double SP_Summaries_welch_rect_area_5_1(const double y[],
                                               const int size);
extern double SP_Summaries_welch_rect_centroid(const double y[],
                                               const int size);

// stats utility — needed for z-score normalization
extern void zscore_norm2(const double a[], const int size, double b[]);
}

// Cached autocorrelation functions — compute autocorrelation once, reuse for
// features 2, 3, 8, 10, 12 (avoids 5 redundant FFT-based autocorr calls).
#include "c22_autocorr_opt.h"

namespace SCAMP {

// ============================================================================
// Internal: Compute a single catch22 feature by index (0..21).
// Used by Level 3 parallelism to dispatch individual features to threads.
// Feature ordering must exactly match compute_c22_features_internal().
// ============================================================================
static double compute_single_feature(int f, const double* z, int n,
                                     const double* autocorrs = nullptr) {
  switch (f) {
    case 0:
      return DN_HistogramMode_5(z, n);
    case 1:
      return DN_HistogramMode_10(z, n);
    case 2:
      return autocorrs ? c22_CO_f1ecac_cached(autocorrs, n) : CO_f1ecac(z, n);
    case 3:
      return autocorrs
                 ? static_cast<double>(c22_CO_FirstMin_ac_cached(autocorrs, n))
                 : static_cast<double>(CO_FirstMin_ac(z, n));
    case 4:
      return CO_HistogramAMI_even_2_5(z, n);
    case 5:
      return CO_trev_1_num(z, n);
    case 6:
      return MD_hrv_classic_pnn40(z, n);
    case 7:
      return SB_BinaryStats_mean_longstretch1(z, n);
    case 8:
      return autocorrs ? c22_SB_TransitionMatrix_3ac_sumdiagcov_cached(
                             z, n, autocorrs)
                       : SB_TransitionMatrix_3ac_sumdiagcov(z, n);
    case 9:
      return static_cast<double>(PD_PeriodicityWang_th0_01(z, n));
    case 10:
      return autocorrs ? c22_CO_Embed2_Dist_tau_d_expfit_meandiff_cached(
                             z, n, autocorrs)
                       : CO_Embed2_Dist_tau_d_expfit_meandiff(z, n);
    case 11:
      return IN_AutoMutualInfoStats_40_gaussian_fmmi(z, n);
    case 12:
      return autocorrs
                 ? c22_FC_LocalSimple_mean1_tauresrat_cached(z, n, autocorrs)
                 : FC_LocalSimple_mean1_tauresrat(z, n);
    case 13:
      return DN_OutlierInclude_p_001_mdrmd(z, n);
    case 14:
      return DN_OutlierInclude_n_001_mdrmd(z, n);
    case 15:
      return SP_Summaries_welch_rect_area_5_1(z, n);
    case 16:
      return SB_BinaryStats_diff_longstretch0(z, n);
    case 17:
      return SB_MotifThree_quantile_hh(z, n);
    case 18:
      return SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(z, n);
    case 19:
      return SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(z, n);
    case 20:
      return SP_Summaries_welch_rect_centroid(z, n);
    case 21:
      return FC_LocalSimple_mean3_stderr(z, n);
    default:
      return 0.0;
  }
}

// ============================================================================
// Internal: Compute all 22 features of ONE subsequence using nthreads threads.
//
// Level 3 parallelism: thread t handles features t, t+nthreads, t+2*nthreads,
// ... (round-robin).  Round-robin naturally spreads expensive features across
// threads: features 13,14 (DN_OutlierInclude) and 18,19 (SC_FluctAnal) land
// on different threads, preventing one thread from monopolising runtime.
//
// ============================================================================
static void compute_c22_features_l3(const double* z, int n,
                                    C22FeatureVector& out, int nthreads) {
  const int NF = C22FeatureVector::NUM_FEATURES;

  // Pre-compute autocorrelation once (FFT-based, O(W log W)) and share it
  // across features 2, 3, 8, 10, 12 — avoids 5 redundant FFT calls.
  double* autocorrs = c22_co_autocorrs(z, n);

  if (nthreads <= 1) {
    for (int f = 0; f < NF; ++f) {
      double v = compute_single_feature(f, z, n, autocorrs);
      out.features[f] = std::isfinite(v) ? v : 0.0;
    }
    free(autocorrs);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(nthreads - 1);

  for (int t = 1; t < nthreads; ++t) {
    workers.emplace_back([t, nthreads, z, n, autocorrs, &out]() {
      for (int f = t; f < C22FeatureVector::NUM_FEATURES; f += nthreads) {
        double v = compute_single_feature(f, z, n, autocorrs);
        out.features[f] = std::isfinite(v) ? v : 0.0;
      }
    });
  }

  for (int f = 0; f < NF; f += nthreads) {
    double v = compute_single_feature(f, z, n, autocorrs);
    out.features[f] = std::isfinite(v) ? v : 0.0;
  }

  for (auto& w : workers) w.join();
  free(autocorrs);
}

C22FeatureVector compute_c22_features_internal(const double* data, int size) {
  C22FeatureVector vec;

  std::vector<double> zscored(size);
  zscore_norm2(data, size, zscored.data());

  compute_c22_features_l3(zscored.data(), size, vec, /*nthreads=*/1);
  return vec;
}

C22FeatureVector compute_c22_features(const std::vector<double>& timeseries,
                                      int start_idx, int window_size) {
  if (start_idx < 0 ||
      static_cast<size_t>(start_idx + window_size) > timeseries.size()) {
    return C22FeatureVector();
  }

  const double* data = timeseries.data() + start_idx;
  return compute_c22_features_internal(data, window_size);
}

// ============================================================================
// Parallel computation of C22 vectors for all N subsequences.
//
// Adaptive two-level dispatch:
//
//   hw = hardware_concurrency()
//   if N >= hw:
//     Level 1 only — hw threads, each processes ceil(N/hw) subsequences
//     sequentially.  All CPU cores stay busy.
//     l1_threads = hw,  l3_threads = 1
//   else (N < hw):
//     Level 1+3 — N threads (one per subsequence); each subsequence uses
//     hw/N feature threads (Level 3) so spare cores are not wasted.
//     l1_threads = N,  l3_threads = max(1, hw/N)
//
// Total threads spawned ≤ l1_threads × l3_threads ≤ hw — no oversubscription.
// ============================================================================
std::vector<C22FeatureVector> compute_c22_vectors_parallel(
    const std::vector<double>& timeseries, int window_size, int num_threads) {
  int num_subsequences = static_cast<int>(timeseries.size()) - window_size + 1;
  if (num_subsequences <= 0) {
    return std::vector<C22FeatureVector>();
  }

  std::vector<C22FeatureVector> result(num_subsequences);

  int hw = num_threads;
  if (hw <= 0) {
    hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw <= 0) hw = 1;
  }

  int l1_threads, l3_threads;
  if (num_subsequences >= hw) {
    l1_threads = hw;
    l3_threads = 1;
  } else {
    l1_threads = num_subsequences;
    l3_threads = std::max(1, hw / num_subsequences);
  }

  std::vector<std::thread> threads;
  int subsequences_per_thread =
      (num_subsequences + l1_threads - 1) / l1_threads;

  for (int t = 0; t < l1_threads; ++t) {
    int start = t * subsequences_per_thread;
    int end = std::min(start + subsequences_per_thread, num_subsequences);

    if (start < num_subsequences) {
      threads.emplace_back(
          [&timeseries, window_size, start, end, &result, l3_threads]() {
            std::vector<double> zscored(window_size);
            for (int i = start; i < end; ++i) {
              const double* data = timeseries.data() + i;
              zscore_norm2(data, window_size, zscored.data());
              compute_c22_features_l3(zscored.data(), window_size, result[i],
                                      l3_threads);
            }
          });
    }
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

}  // namespace SCAMP
