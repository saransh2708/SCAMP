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

namespace SCAMP {

C22FeatureVector compute_c22_features_internal(const double* data, int size) {
  C22FeatureVector vec;

  // Z-score normalize the data first, as required by catch22
  // (see pycatch22/src/C/main.c line 70: zscore_norm2(y, size, y_zscored))
  std::vector<double> zscored(size);
  zscore_norm2(data, size, zscored.data());
  const double* z = zscored.data();

  // Compute all 22 catch22 features in order
  vec.features[0] = DN_HistogramMode_5(z, size);
  vec.features[1] = DN_HistogramMode_10(z, size);
  vec.features[2] = CO_f1ecac(z, size);
  vec.features[3] = static_cast<double>(CO_FirstMin_ac(z, size));
  vec.features[4] = CO_HistogramAMI_even_2_5(z, size);
  vec.features[5] = CO_trev_1_num(z, size);
  vec.features[6] = MD_hrv_classic_pnn40(z, size);
  vec.features[7] = SB_BinaryStats_mean_longstretch1(z, size);
  vec.features[8] = SB_TransitionMatrix_3ac_sumdiagcov(z, size);
  vec.features[9] = static_cast<double>(PD_PeriodicityWang_th0_01(z, size));
  vec.features[10] = CO_Embed2_Dist_tau_d_expfit_meandiff(z, size);
  vec.features[11] = IN_AutoMutualInfoStats_40_gaussian_fmmi(z, size);
  vec.features[12] = FC_LocalSimple_mean1_tauresrat(z, size);
  vec.features[13] = DN_OutlierInclude_p_001_mdrmd(z, size);
  vec.features[14] = DN_OutlierInclude_n_001_mdrmd(z, size);
  vec.features[15] = SP_Summaries_welch_rect_area_5_1(z, size);
  vec.features[16] = SB_BinaryStats_diff_longstretch0(z, size);
  vec.features[17] = SB_MotifThree_quantile_hh(z, size);
  vec.features[18] = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(z, size);
  vec.features[19] = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(z, size);
  vec.features[20] = SP_Summaries_welch_rect_centroid(z, size);
  vec.features[21] = FC_LocalSimple_mean3_stderr(z, size);

  // Replace NaN/Inf values with 0.0 for safe dot product computation
  for (int i = 0; i < C22FeatureVector::NUM_FEATURES; ++i) {
    if (!std::isfinite(vec.features[i])) {
      vec.features[i] = 0.0;
    }
  }

  return vec;
}

C22FeatureVector compute_c22_features(const std::vector<double>& timeseries,
                                      int start_idx, int window_size) {
  if (start_idx < 0 ||
      static_cast<size_t>(start_idx + window_size) > timeseries.size()) {
    // Return zero vector if out of bounds
    return C22FeatureVector();
  }

  const double* data = timeseries.data() + start_idx;
  return compute_c22_features_internal(data, window_size);
}

// Parallel computation of C22 vectors for all subsequences
std::vector<C22FeatureVector> compute_c22_vectors_parallel(
    const std::vector<double>& timeseries, int window_size, int num_threads) {
  int num_subsequences = static_cast<int>(timeseries.size()) - window_size + 1;
  if (num_subsequences <= 0) {
    return std::vector<C22FeatureVector>();
  }

  std::vector<C22FeatureVector> result(num_subsequences);

  // Auto-detect number of threads; fall back to 1 if detection fails
  if (num_threads <= 0) {
    num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0) {
      num_threads = 1;
    }
  }

  // Cap threads to number of subsequences (no point spawning idle threads)
  if (num_threads > num_subsequences) {
    num_threads = num_subsequences;
  }

  // Parallel computation: each thread processes a range of subsequences
  std::vector<std::thread> threads;
  int subsequences_per_thread =
      (num_subsequences + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start = t * subsequences_per_thread;
    int end = std::min(start + subsequences_per_thread, num_subsequences);

    if (start < num_subsequences) {
      threads.emplace_back([&timeseries, window_size, start, end, &result]() {
        for (int i = start; i < end; ++i) {
          result[i] = compute_c22_features(timeseries, i, window_size);
        }
      });
    }
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}

}  // namespace SCAMP
