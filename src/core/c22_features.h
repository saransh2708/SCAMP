#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace SCAMP {

// C22 Feature Vector: Contains all 22 catch22 features.
// Uses std::array (stack-allocated, 22×8 = 176 bytes) so that the N
// intermediate C22FeatureVector objects created during feature extraction
// require zero heap allocations, keeping allocator pressure off the hot path.
struct C22FeatureVector {
  static constexpr int NUM_FEATURES = 22;
  std::array<double, NUM_FEATURES> features;

  C22FeatureVector() { features.fill(0.0); }

  // Compute dot product with another C22 feature vector
  double dot_product(const C22FeatureVector& other) const {
    double result = 0.0;
    for (int i = 0; i < NUM_FEATURES; ++i) {
      result += features[i] * other.features[i];
    }
    return result;
  }
};

// Compute C22 feature vector for a raw subsequence (z-scores internally)
// Input: time series data, start index, window size (subsequence length)
C22FeatureVector compute_c22_features(const std::vector<double>& timeseries,
                                      int start_idx, int window_size);

// Compute C22 feature vectors for all subsequences in parallel
// Returns vector of C22FeatureVector, one per subsequence
std::vector<C22FeatureVector> compute_c22_vectors_parallel(
    const std::vector<double>& timeseries, int window_size,
    int num_threads = 0);  // 0 = auto-detect

// Compute C22 feature vector from raw data pointer (z-scores internally)
C22FeatureVector compute_c22_features_internal(const double* data, int size);

}  // namespace SCAMP
