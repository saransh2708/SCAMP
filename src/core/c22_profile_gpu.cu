#ifdef _HAS_CUDA_

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// Number of catch22 features
#define C22_NUM_FEATURES 22

// Block size for the dot product kernel
#define C22_BLOCK_SIZE 256

// CUDA error checking macro (matches SCAMP's gpuErrchk pattern)
#define C22_CUDA_CHECK(ans)                     \
  {                                             \
    c22_cuda_assert((ans), __FILE__, __LINE__); \
  }

static void c22_cuda_assert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "C22 CUDA Error: %s  %s:%d\n", cudaGetErrorString(code),
            file, line);
    exit(code);
  }
}

// ============================================================================
// CUDA Kernel: For each row i, find the column j that maximizes the dot
// product of F[i] · F[j], subject to exclusion zone |i - j| > exclusion.
//
// Grid:  N blocks (one per row)
// Block: C22_BLOCK_SIZE threads
// Each thread scans a strided subset of columns.
// ============================================================================
__global__ void c22_profile_kernel(
    const double* __restrict__ F,  // Flattened feature matrix: N x 22
    double* __restrict__ profile,  // Output: max dot product per row
    int* __restrict__ index,       // Output: index of best match per row
    int N,                         // Number of subsequences
    int exclusion) {               // Exclusion zone radius

  const int row = blockIdx.x;
  if (row >= N) return;

  // Load this row's feature vector into shared memory
  __shared__ double row_vec[C22_NUM_FEATURES];
  if (threadIdx.x < C22_NUM_FEATURES) {
    row_vec[threadIdx.x] = F[row * C22_NUM_FEATURES + threadIdx.x];
  }
  __syncthreads();

  // Each thread finds its local best match
  double local_best_dot = -DBL_MAX;
  int local_best_j = -1;

  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    // Exclusion zone check
    int diff = (row > j) ? (row - j) : (j - row);
    if (diff <= exclusion) continue;

    // Compute dot product
    double dot = 0.0;
    const double* col_vec = &F[j * C22_NUM_FEATURES];
    for (int f = 0; f < C22_NUM_FEATURES; ++f) {
      dot += row_vec[f] * col_vec[f];
    }

    if (dot > local_best_dot) {
      local_best_dot = dot;
      local_best_j = j;
    }
  }

  // Block-level reduction: find the maximum across all threads
  // Using shared memory reduction
  __shared__ double s_dots[C22_BLOCK_SIZE];
  __shared__ int s_idxs[C22_BLOCK_SIZE];
  s_dots[threadIdx.x] = local_best_dot;
  s_idxs[threadIdx.x] = local_best_j;
  __syncthreads();

  // Standard parallel reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_dots[threadIdx.x + stride] > s_dots[threadIdx.x]) {
        s_dots[threadIdx.x] = s_dots[threadIdx.x + stride];
        s_idxs[threadIdx.x] = s_idxs[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  // Thread 0 writes the result
  if (threadIdx.x == 0) {
    profile[row] = s_dots[0];
    index[row] = s_idxs[0];
  }
}

// ============================================================================
// AB-join kernel: For each row i in A, find column j in B that maximizes
// F_A[i] · F_B[j]. No exclusion zone needed.
// ============================================================================
__global__ void c22_profile_abjoin_kernel(
    const double* __restrict__ FA,  // Feature matrix A: NA x 22
    const double* __restrict__ FB,  // Feature matrix B: NB x 22
    double* __restrict__ profile,   // Output: max dot product per row of A
    int* __restrict__ index,        // Output: index of best match in B
    int NA,                         // Number of subsequences in A
    int NB) {                       // Number of subsequences in B

  const int row = blockIdx.x;
  if (row >= NA) return;

  // Load A's feature vector into shared memory
  __shared__ double row_vec[C22_NUM_FEATURES];
  if (threadIdx.x < C22_NUM_FEATURES) {
    row_vec[threadIdx.x] = FA[row * C22_NUM_FEATURES + threadIdx.x];
  }
  __syncthreads();

  // Each thread scans a strided subset of B
  double local_best_dot = -DBL_MAX;
  int local_best_j = -1;

  for (int j = threadIdx.x; j < NB; j += blockDim.x) {
    double dot = 0.0;
    const double* col_vec = &FB[j * C22_NUM_FEATURES];
    for (int f = 0; f < C22_NUM_FEATURES; ++f) {
      dot += row_vec[f] * col_vec[f];
    }
    if (dot > local_best_dot) {
      local_best_dot = dot;
      local_best_j = j;
    }
  }

  // Block-level reduction
  __shared__ double s_dots[C22_BLOCK_SIZE];
  __shared__ int s_idxs[C22_BLOCK_SIZE];
  s_dots[threadIdx.x] = local_best_dot;
  s_idxs[threadIdx.x] = local_best_j;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_dots[threadIdx.x + stride] > s_dots[threadIdx.x]) {
        s_dots[threadIdx.x] = s_dots[threadIdx.x + stride];
        s_idxs[threadIdx.x] = s_idxs[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    profile[row] = s_dots[0];
    index[row] = s_idxs[0];
  }
}

// ============================================================================
// Host-side wrappers (called from c22_profile.cpp)
// ============================================================================

extern "C" {

void c22_profile_selfjoin_gpu_launch(
    const double* h_features,  // Host: flattened N x 22
    double* h_profile,         // Host: output profile (N elements)
    int* h_index,              // Host: output indices (N elements)
    int N,                     // Number of subsequences
    int exclusion,             // Exclusion zone
    int gpu_id) {              // GPU device ID

  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_bytes = (size_t)N * C22_NUM_FEATURES * sizeof(double);
  size_t prof_bytes = (size_t)N * sizeof(double);
  size_t idx_bytes = (size_t)N * sizeof(int);

  // Allocate device memory
  double* d_features = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  // Copy features to device
  C22_CUDA_CHECK(
      cudaMemcpy(d_features, h_features, feat_bytes, cudaMemcpyHostToDevice));

  // Launch kernel
  int grid_size = N;
  int block_size = C22_BLOCK_SIZE;
  c22_profile_kernel<<<grid_size, block_size>>>(d_features, d_profile, d_index,
                                                N, exclusion);
  C22_CUDA_CHECK(cudaGetLastError());       // check launch errors
  C22_CUDA_CHECK(cudaDeviceSynchronize());  // wait for kernel completion

  // Copy results back
  C22_CUDA_CHECK(
      cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  C22_CUDA_CHECK(
      cudaMemcpy(h_index, d_index, idx_bytes, cudaMemcpyDeviceToHost));

  // Free device memory
  C22_CUDA_CHECK(cudaFree(d_features));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

void c22_profile_abjoin_gpu_launch(
    const double* h_features_a,  // Host: flattened NA x 22
    const double* h_features_b,  // Host: flattened NB x 22
    double* h_profile,           // Host: output profile (NA elements)
    int* h_index,                // Host: output indices (NA elements)
    int NA,                      // Number of subsequences in A
    int NB,                      // Number of subsequences in B
    int gpu_id) {                // GPU device ID

  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_a_bytes = (size_t)NA * C22_NUM_FEATURES * sizeof(double);
  size_t feat_b_bytes = (size_t)NB * C22_NUM_FEATURES * sizeof(double);
  size_t prof_bytes = (size_t)NA * sizeof(double);
  size_t idx_bytes = (size_t)NA * sizeof(int);

  double* d_features_a = nullptr;
  double* d_features_b = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features_a, feat_a_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_features_b, feat_b_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  C22_CUDA_CHECK(cudaMemcpy(d_features_a, h_features_a, feat_a_bytes,
                            cudaMemcpyHostToDevice));
  C22_CUDA_CHECK(cudaMemcpy(d_features_b, h_features_b, feat_b_bytes,
                            cudaMemcpyHostToDevice));

  int grid_size = NA;
  int block_size = C22_BLOCK_SIZE;
  c22_profile_abjoin_kernel<<<grid_size, block_size>>>(
      d_features_a, d_features_b, d_profile, d_index, NA, NB);
  C22_CUDA_CHECK(cudaGetLastError());
  C22_CUDA_CHECK(cudaDeviceSynchronize());

  C22_CUDA_CHECK(
      cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  C22_CUDA_CHECK(
      cudaMemcpy(h_index, d_index, idx_bytes, cudaMemcpyDeviceToHost));

  C22_CUDA_CHECK(cudaFree(d_features_a));
  C22_CUDA_CHECK(cudaFree(d_features_b));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

}  // extern "C"

#endif  // _HAS_CUDA_
