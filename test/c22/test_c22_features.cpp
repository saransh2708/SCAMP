#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "../src/core/c22_features.h"

using namespace SCAMP;

int main() {
  std::cout << "=== C22 Feature Integration Test ===\n\n";

  // --- Test 1: Basic feature vector computation ---
  std::cout << "Test 1: Single feature vector computation\n";
  std::vector<double> ts;
  for (int i = 0; i < 200; ++i) {
    ts.push_back(std::sin(2.0 * M_PI * i / 50.0) + 0.1 * (i % 7));
  }

  C22FeatureVector vec = compute_c22_features(ts, 0, 100);
  assert(vec.features.size() == 22);

  int non_zero = 0;
  for (int i = 0; i < 22; ++i) {
    assert(std::isfinite(vec.features[i]));
    if (std::abs(vec.features[i]) > 1e-12) non_zero++;
  }
  std::cout << "  22 features computed, " << non_zero << " non-zero\n";
  assert(non_zero > 0 && "At least some features should be non-zero");
  std::cout << "  PASSED\n\n";

  // --- Test 2: Z-score invariance ---
  // Shifting/scaling the input should NOT change feature values
  // because z-scoring normalizes the data first
  std::cout << "Test 2: Z-score normalization (shift/scale invariance)\n";
  std::vector<double> ts_shifted(ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    ts_shifted[i] = ts[i] * 3.0 + 42.0;  // scale by 3, shift by 42
  }

  C22FeatureVector vec_orig = compute_c22_features(ts, 0, 100);
  C22FeatureVector vec_shifted = compute_c22_features(ts_shifted, 0, 100);

  double max_diff = 0.0;
  for (int i = 0; i < 22; ++i) {
    double diff = std::abs(vec_orig.features[i] - vec_shifted.features[i]);
    if (diff > max_diff) max_diff = diff;
  }
  std::cout << "  Max feature diff after shift+scale: " << max_diff << "\n";
  assert(max_diff < 1e-8 &&
         "Features must be invariant to shift+scale (z-scored)");
  std::cout << "  PASSED\n\n";

  // --- Test 3: Dot product ---
  std::cout << "Test 3: Dot product\n";
  C22FeatureVector v1 = compute_c22_features(ts, 0, 100);
  C22FeatureVector v2 = compute_c22_features(ts, 50, 100);
  double self_dot = v1.dot_product(v1);
  double cross_dot = v1.dot_product(v2);
  std::cout << "  Self dot product:  " << self_dot << "\n";
  std::cout << "  Cross dot product: " << cross_dot << "\n";
  assert(std::isfinite(self_dot) && self_dot >= 0);
  assert(std::isfinite(cross_dot));
  std::cout << "  PASSED\n\n";

  // --- Test 4: Out-of-bounds returns zero vector ---
  std::cout << "Test 4: Out-of-bounds safety\n";
  C22FeatureVector vec_oob = compute_c22_features(ts, 999, 100);
  for (int i = 0; i < 22; ++i) {
    assert(vec_oob.features[i] == 0.0);
  }
  std::cout << "  PASSED\n\n";

  // --- Test 5: Parallel computation ---
  std::cout << "Test 5: Parallel computation\n";
  int window = 100;
  std::vector<C22FeatureVector> vectors =
      compute_c22_vectors_parallel(ts, window, 4);
  int expected_count = static_cast<int>(ts.size()) - window + 1;
  assert(static_cast<int>(vectors.size()) == expected_count);

  // Verify consistency: parallel result[0] == single compute at index 0
  C22FeatureVector vec_single = compute_c22_features(ts, 0, window);
  double consistency_diff = 0.0;
  for (int i = 0; i < 22; ++i) {
    double d = std::abs(vectors[0].features[i] - vec_single.features[i]);
    if (d > consistency_diff) consistency_diff = d;
  }
  std::cout << "  " << vectors.size() << " vectors computed\n";
  std::cout << "  Parallel vs single max diff: " << consistency_diff << "\n";
  assert(consistency_diff < 1e-12);
  std::cout << "  PASSED\n\n";

  // --- Test 6: PD_PeriodicityWang_th0_01 returns integer value ---
  std::cout << "Test 6: PD_PeriodicityWang_th0_01 (feature[9]) is integer\n";
  double pd_val = vec.features[9];
  assert(std::isfinite(pd_val));
  assert(pd_val == std::floor(pd_val) && "PD feature should be integer-valued");
  std::cout << "  Value: " << pd_val << " (integer: yes)\n";
  std::cout << "  PASSED\n\n";

  std::cout << "=== ALL TESTS PASSED ===\n";
  return 0;
}
