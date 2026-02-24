#pragma once

#include <vector>

namespace SCAMP {

// Result of a C22 Profile computation
struct C22ProfileResult {
  // For each subsequence: the maximum C22 dot product similarity
  std::vector<double> profile;
  // For each subsequence: the index of the best matching subsequence
  std::vector<int> index;
};

// ============================================================================
// CPU Implementation (multi-threaded)
// ============================================================================

// Self-join: For each subsequence of T, find the most similar OTHER
// subsequence (by C22 feature vector dot product), respecting exclusion zone.
//
//   profile[i] = max over j (where |i-j| > exclusion) of F_i · F_j
//   index[i]   = argmax j
C22ProfileResult c22_profile_selfjoin_cpu(const std::vector<double>& timeseries,
                                          int window_size, int num_threads = 0);

// AB-join: For each subsequence of A, find the most similar subsequence in B.
C22ProfileResult c22_profile_abjoin_cpu(const std::vector<double>& timeseries_a,
                                        const std::vector<double>& timeseries_b,
                                        int window_size, int num_threads = 0);

// ============================================================================
// GPU Implementation (CUDA)
// ============================================================================

// Self-join on GPU. Feature computation runs on CPU; dot product search on GPU.
C22ProfileResult c22_profile_selfjoin_gpu(const std::vector<double>& timeseries,
                                          int window_size, int gpu_id = 0,
                                          int num_cpu_threads = 0);

// AB-join on GPU.
C22ProfileResult c22_profile_abjoin_gpu(const std::vector<double>& timeseries_a,
                                        const std::vector<double>& timeseries_b,
                                        int window_size, int gpu_id = 0,
                                        int num_cpu_threads = 0);

}  // namespace SCAMP
