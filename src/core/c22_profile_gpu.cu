#ifdef _HAS_CUDA_

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

// Number of catch22 features
#define C22_NUM_FEATURES 22

// Declared in c22_features_gpu.cu (compiled into the same library)
#define C22_GPU_BLOCK_SIZE 64
#define C22_GPU_MAX_W 256
__global__ void c22_compute_features_kernel(const double* __restrict__, int,
                                            int, int, double* __restrict__);

// Tile dimensions for the tiled dot-product kernel.
// Each CUDA block handles a (TILE_R x TILE_C) region of the N x N similarity
// matrix.  Both tiles are staged in shared memory so each feature vector is
// loaded from global memory only once per tile rather than once per cell.
//
//   2 * 32 * 22 * 8 = 11,264 bytes shared memory (well within 48 KB limit)
#define TILE_R 32
#define TILE_C 32

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
// Packed atomicMax trick — atomically updates (profile_value, index) together.
//
// Encoding: pack float(dot) and int(col) into a single uint64:
//   high 32 bits = monotone bit-encoding of float(dot)   ← drives comparison
//   low  32 bits = uint32(col)                            ← stored index
//
// A single atomicMax on the uint64 atomically keeps the cell with the largest
// dot product (and, for exact ties, the largest column index, which is fine
// since any valid nearest-neighbor index is acceptable).
//
// Monotone float encoding (standard "flip" trick from GPU sorting literature):
//   positive floats: flip sign bit   → 1xxxxxxx...  (larger uint → larger
//   float) negative floats: flip all bits   → 0xxxxxxx...  (larger uint → less
//   negative)
// ============================================================================
__device__ static inline unsigned int float_to_monotone_bits(float v) {
  unsigned int bits;
  // Use memcpy to avoid strict-aliasing UB; nvcc optimises this away.
  __builtin_memcpy(&bits, &v, sizeof(float));
  if (bits >> 31u) {  // negative: flip all bits
    bits = ~bits;
  } else {  // non-negative: set sign bit
    bits |= 0x80000000u;
  }
  return bits;
}

__device__ static inline unsigned long long pack_dot_idx(float dot, int j) {
  return ((unsigned long long)float_to_monotone_bits(dot) << 32) |
         (unsigned long long)(unsigned int)j;
}

// ============================================================================
// Second-pass kernel: given the winning index for each row, recompute the
// exact double-precision dot product so the final profile is exact.
// Grid: ceil(N/256) blocks × 256 threads (1-D, one thread per row).
// ============================================================================
__global__ void c22_recompute_profile_kernel(const double* __restrict__ F,
                                             const int* __restrict__ idx,
                                             double* __restrict__ profile,
                                             int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || idx[row] < 0) return;
  int j = idx[row];
  double dot = 0.0;
  const double* fi = F + (size_t)row * C22_NUM_FEATURES;
  const double* fj = F + (size_t)j * C22_NUM_FEATURES;
  for (int f = 0; f < C22_NUM_FEATURES; ++f) dot += fi[f] * fj[f];
  profile[row] = dot;
}

// AB-join variant of the recompute pass (FA rows, FB columns)
__global__ void c22_recompute_abjoin_profile_kernel(
    const double* __restrict__ FA, const double* __restrict__ FB,
    const int* __restrict__ idx, double* __restrict__ profile, int NA) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= NA || idx[row] < 0) return;
  int j = idx[row];
  double dot = 0.0;
  const double* fi = FA + (size_t)row * C22_NUM_FEATURES;
  const double* fj = FB + (size_t)j * C22_NUM_FEATURES;
  for (int f = 0; f < C22_NUM_FEATURES; ++f) dot += fi[f] * fj[f];
  profile[row] = dot;
}

// ============================================================================
// Tiled self-join kernel
//
// Grid:  (ceil(N/TILE_C), ceil(N/TILE_R))  2-D grid of tiles
// Block: (TILE_C, TILE_R)                  one thread per cell in the tile
//
// Each block cooperatively loads TILE_R row-vectors and TILE_C col-vectors
// into shared memory, then each thread computes one dot product using those
// cached values.  A single 64-bit atomicMax on packed[row] atomically keeps
// the (float_profile, index) pair with the largest dot product for that row.
// ============================================================================
__global__ void c22_tiled_selfjoin_kernel(
    const double* __restrict__ F,             // N x 22  (row-major)
    unsigned long long* __restrict__ packed,  // N — packed (float_dot, idx)
    int N, int exclusion) {
  __shared__ double sh_row[TILE_R][C22_NUM_FEATURES];
  __shared__ double sh_col[TILE_C][C22_NUM_FEATURES];

  const int tx = threadIdx.x;  // 0..TILE_C-1  → column within tile
  const int ty = threadIdx.y;  // 0..TILE_R-1  → row within tile

  const int row = blockIdx.y * TILE_R + ty;
  const int col = blockIdx.x * TILE_C + tx;

  // ---- Cooperative load: row vectors ----
  {
    int load_row = blockIdx.y * TILE_R + ty;
    for (int f = tx; f < C22_NUM_FEATURES; f += TILE_C)
      sh_row[ty][f] = (load_row < N) ? F[load_row * C22_NUM_FEATURES + f] : 0.0;
  }

  // ---- Cooperative load: column vectors ----
  {
    int load_col = blockIdx.x * TILE_C + tx;
    for (int f = ty; f < C22_NUM_FEATURES; f += TILE_R)
      sh_col[tx][f] = (load_col < N) ? F[load_col * C22_NUM_FEATURES + f] : 0.0;
  }

  __syncthreads();

  // ---- Compute dot product and atomically update packed[row] ----
  if (row < N && col < N) {
    int diff = (row > col) ? (row - col) : (col - row);
    if (diff > exclusion) {
      double dot = 0.0;
      for (int f = 0; f < C22_NUM_FEATURES; ++f)
        dot += sh_row[ty][f] * sh_col[tx][f];

      // Single atomic: (float_dot, col) → atomically wins if dot is largest
      atomicMax(&packed[row], pack_dot_idx((float)dot, col));
    }
  }
}

// ============================================================================
// Tiled AB-join kernel  (same structure, no exclusion zone)
// ============================================================================
__global__ void c22_tiled_abjoin_kernel(
    const double* __restrict__ FA,            // NA x 22
    const double* __restrict__ FB,            // NB x 22
    unsigned long long* __restrict__ packed,  // NA — packed (float_dot, idx)
    int NA, int NB) {
  __shared__ double sh_row[TILE_R][C22_NUM_FEATURES];
  __shared__ double sh_col[TILE_C][C22_NUM_FEATURES];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * TILE_R + ty;
  const int col = blockIdx.x * TILE_C + tx;

  {
    int load_row = blockIdx.y * TILE_R + ty;
    for (int f = tx; f < C22_NUM_FEATURES; f += TILE_C)
      sh_row[ty][f] =
          (load_row < NA) ? FA[load_row * C22_NUM_FEATURES + f] : 0.0;
  }
  {
    int load_col = blockIdx.x * TILE_C + tx;
    for (int f = ty; f < C22_NUM_FEATURES; f += TILE_R)
      sh_col[tx][f] =
          (load_col < NB) ? FB[load_col * C22_NUM_FEATURES + f] : 0.0;
  }

  __syncthreads();

  if (row < NA && col < NB) {
    double dot = 0.0;
    for (int f = 0; f < C22_NUM_FEATURES; ++f)
      dot += sh_row[ty][f] * sh_col[tx][f];

    atomicMax(&packed[row], pack_dot_idx((float)dot, col));
  }
}

// ============================================================================
// Helper: initial "sentinel" packed value = (float(-DBL_MAX), idx=-1).
// This is the smallest possible packed value, so any real dot product wins.
// ============================================================================
static unsigned long long sentinel_packed() {
  // float(-DBL_MAX) is extremely negative; monotone encoding → 0x00100000
  // (smallest positive uint32 after encoding, because ~bits of a very negative
  // float is a very small uint).  Combined with idx 0xFFFFFFFF (-1 as uint),
  // the full 64-bit value is smaller than any valid packed dot product.
  float neg_max = -FLT_MAX;
  unsigned int bits;
  __builtin_memcpy(&bits, &neg_max, sizeof(float));
  bits = ~bits;  // negative: flip all bits
  return ((unsigned long long)bits << 32) | 0xFFFFFFFFULL;
}

// ============================================================================
// Host-side wrappers (called from c22_profile.cpp)
// ============================================================================

extern "C" {

void c22_profile_selfjoin_gpu_launch(const double* h_features,
                                     double* h_profile, int* h_index, int N,
                                     int exclusion, int gpu_id) {
  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_bytes = (size_t)N * C22_NUM_FEATURES * sizeof(double);
  size_t packed_bytes = (size_t)N * sizeof(unsigned long long);
  size_t prof_bytes = (size_t)N * sizeof(double);
  size_t idx_bytes = (size_t)N * sizeof(int);

  double* d_features = nullptr;
  unsigned long long* d_packed = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  // Initialise packed[] with sentinel so any real dot product wins atomicMax
  {
    unsigned long long s = sentinel_packed();
    std::vector<unsigned long long> init(N, s);
    C22_CUDA_CHECK(cudaMemcpy(d_packed, init.data(), packed_bytes,
                              cudaMemcpyHostToDevice));
  }

  C22_CUDA_CHECK(
      cudaMemcpy(d_features, h_features, feat_bytes, cudaMemcpyHostToDevice));

  // Phase 1: tiled dot-product kernel — fills d_packed atomically
  dim3 block(TILE_C, TILE_R);
  dim3 grid((N + TILE_C - 1) / TILE_C, (N + TILE_R - 1) / TILE_R);
  c22_tiled_selfjoin_kernel<<<grid, block>>>(d_features, d_packed, N,
                                             exclusion);
  C22_CUDA_CHECK(cudaGetLastError());
  C22_CUDA_CHECK(cudaDeviceSynchronize());

  // Decode packed → d_index (CPU-side, O(N))
  {
    std::vector<unsigned long long> h_packed(N);
    C22_CUDA_CHECK(cudaMemcpy(h_packed.data(), d_packed, packed_bytes,
                              cudaMemcpyDeviceToHost));
    std::vector<int> h_idx(N);
    for (int i = 0; i < N; ++i)
      h_idx[i] = (int)(unsigned int)(h_packed[i] & 0xFFFFFFFFULL);
    C22_CUDA_CHECK(
        cudaMemcpy(d_index, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    // Also copy h_idx to output now; profile recomputed next
    std::copy(h_idx.begin(), h_idx.end(), h_index);
  }

  // Phase 2: recompute exact double profile from winning indices
  {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    c22_recompute_profile_kernel<<<blocks, threads>>>(d_features, d_index,
                                                      d_profile, N);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
    C22_CUDA_CHECK(
        cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  }

  C22_CUDA_CHECK(cudaFree(d_features));
  C22_CUDA_CHECK(cudaFree(d_packed));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

void c22_profile_abjoin_gpu_launch(const double* h_features_a,
                                   const double* h_features_b,
                                   double* h_profile, int* h_index, int NA,
                                   int NB, int gpu_id) {
  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_a_bytes = (size_t)NA * C22_NUM_FEATURES * sizeof(double);
  size_t feat_b_bytes = (size_t)NB * C22_NUM_FEATURES * sizeof(double);
  size_t packed_bytes = (size_t)NA * sizeof(unsigned long long);
  size_t prof_bytes = (size_t)NA * sizeof(double);
  size_t idx_bytes = (size_t)NA * sizeof(int);

  double* d_features_a = nullptr;
  double* d_features_b = nullptr;
  unsigned long long* d_packed = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features_a, feat_a_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_features_b, feat_b_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_packed, packed_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  {
    unsigned long long s = sentinel_packed();
    std::vector<unsigned long long> init(NA, s);
    C22_CUDA_CHECK(cudaMemcpy(d_packed, init.data(), packed_bytes,
                              cudaMemcpyHostToDevice));
  }

  C22_CUDA_CHECK(cudaMemcpy(d_features_a, h_features_a, feat_a_bytes,
                            cudaMemcpyHostToDevice));
  C22_CUDA_CHECK(cudaMemcpy(d_features_b, h_features_b, feat_b_bytes,
                            cudaMemcpyHostToDevice));

  dim3 block(TILE_C, TILE_R);
  dim3 grid((NB + TILE_C - 1) / TILE_C, (NA + TILE_R - 1) / TILE_R);
  c22_tiled_abjoin_kernel<<<grid, block>>>(d_features_a, d_features_b, d_packed,
                                           NA, NB);
  C22_CUDA_CHECK(cudaGetLastError());
  C22_CUDA_CHECK(cudaDeviceSynchronize());

  // Decode packed → d_index
  {
    std::vector<unsigned long long> h_packed(NA);
    C22_CUDA_CHECK(cudaMemcpy(h_packed.data(), d_packed, packed_bytes,
                              cudaMemcpyDeviceToHost));
    std::vector<int> h_idx(NA);
    for (int i = 0; i < NA; ++i)
      h_idx[i] = (int)(unsigned int)(h_packed[i] & 0xFFFFFFFFULL);
    C22_CUDA_CHECK(
        cudaMemcpy(d_index, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    std::copy(h_idx.begin(), h_idx.end(), h_index);
  }

  // Recompute exact double profile
  {
    int threads = 256;
    int blocks = (NA + threads - 1) / threads;
    c22_recompute_abjoin_profile_kernel<<<blocks, threads>>>(
        d_features_a, d_features_b, d_index, d_profile, NA);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
    C22_CUDA_CHECK(
        cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  }

  C22_CUDA_CHECK(cudaFree(d_features_a));
  C22_CUDA_CHECK(cudaFree(d_features_b));
  C22_CUDA_CHECK(cudaFree(d_packed));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

// ============================================================================
// All-GPU pipeline: raw timeseries → features on GPU → dot product on GPU.
// Called when window ≤ C22_GPU_MAX_W (256).  For larger windows the caller
// falls back to CPU feature extraction + c22_profile_selfjoin_gpu_launch.
// ============================================================================

void c22_profile_selfjoin_gpu_launch_from_ts(const double* h_ts, int ts_length,
                                             int window, double* h_profile,
                                             int* h_index, int N, int exclusion,
                                             int gpu_id) {
  C22_CUDA_CHECK(cudaSetDevice(gpu_id));
  // 128 KB stack: actual kernel usage is ~58 KB (DN_OutlierInclude dominates)
  C22_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024));

  size_t ts_bytes = (size_t)ts_length * sizeof(double);
  size_t feat_bytes = (size_t)N * C22_NUM_FEATURES * sizeof(double);
  size_t pack_bytes = (size_t)N * sizeof(unsigned long long);
  size_t prof_bytes = (size_t)N * sizeof(double);
  size_t idx_bytes = (size_t)N * sizeof(int);

  double* d_ts = nullptr;
  double* d_features = nullptr;
  unsigned long long* d_packed = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_ts, ts_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_packed, pack_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  C22_CUDA_CHECK(cudaMemcpy(d_ts, h_ts, ts_bytes, cudaMemcpyHostToDevice));

  // Phase 1a: compute all N×22 features entirely on GPU
  {
    int blk = C22_GPU_BLOCK_SIZE;
    int grid = (N + blk - 1) / blk;
    c22_compute_features_kernel<<<grid, blk>>>(d_ts, ts_length, window, N,
                                               d_features);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Phase 1b: initialise packed sentinel
  {
    unsigned long long s = sentinel_packed();
    std::vector<unsigned long long> init(N, s);
    C22_CUDA_CHECK(
        cudaMemcpy(d_packed, init.data(), pack_bytes, cudaMemcpyHostToDevice));
  }

  // Phase 2: tiled dot-product kernel
  {
    dim3 block(TILE_C, TILE_R);
    dim3 grid((N + TILE_C - 1) / TILE_C, (N + TILE_R - 1) / TILE_R);
    c22_tiled_selfjoin_kernel<<<grid, block>>>(d_features, d_packed, N,
                                               exclusion);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Phase 3: decode packed → d_index (O(N) on host)
  {
    std::vector<unsigned long long> h_packed(N);
    C22_CUDA_CHECK(cudaMemcpy(h_packed.data(), d_packed, pack_bytes,
                              cudaMemcpyDeviceToHost));
    std::vector<int> h_idx(N);
    for (int i = 0; i < N; ++i)
      h_idx[i] = (int)(unsigned int)(h_packed[i] & 0xFFFFFFFFULL);
    C22_CUDA_CHECK(
        cudaMemcpy(d_index, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    std::copy(h_idx.begin(), h_idx.end(), h_index);
  }

  // Phase 4: exact double-precision profile recompute
  {
    int thd = 256, blk = (N + thd - 1) / thd;
    c22_recompute_profile_kernel<<<blk, thd>>>(d_features, d_index, d_profile,
                                               N);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
    C22_CUDA_CHECK(
        cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  }

  C22_CUDA_CHECK(cudaFree(d_ts));
  C22_CUDA_CHECK(cudaFree(d_features));
  C22_CUDA_CHECK(cudaFree(d_packed));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

void c22_profile_abjoin_gpu_launch_from_ts(const double* h_ts_a,
                                           int ts_a_length,
                                           const double* h_ts_b,
                                           int ts_b_length, int window,
                                           double* h_profile, int* h_index,
                                           int NA, int NB, int gpu_id) {
  C22_CUDA_CHECK(cudaSetDevice(gpu_id));
  // 128 KB stack: actual kernel usage is ~58 KB (DN_OutlierInclude dominates)
  C22_CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 128 * 1024));

  size_t ts_a_bytes = (size_t)ts_a_length * sizeof(double);
  size_t ts_b_bytes = (size_t)ts_b_length * sizeof(double);
  size_t feat_a_bytes = (size_t)NA * C22_NUM_FEATURES * sizeof(double);
  size_t feat_b_bytes = (size_t)NB * C22_NUM_FEATURES * sizeof(double);
  size_t pack_bytes = (size_t)NA * sizeof(unsigned long long);
  size_t prof_bytes = (size_t)NA * sizeof(double);
  size_t idx_bytes = (size_t)NA * sizeof(int);

  double* d_ts_a = nullptr;
  double* d_ts_b = nullptr;
  double* d_feat_a = nullptr;
  double* d_feat_b = nullptr;
  unsigned long long* d_packed = nullptr;
  double* d_profile = nullptr;
  int* d_index = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_ts_a, ts_a_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_ts_b, ts_b_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_feat_a, feat_a_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_feat_b, feat_b_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_packed, pack_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile, prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index, idx_bytes));

  C22_CUDA_CHECK(
      cudaMemcpy(d_ts_a, h_ts_a, ts_a_bytes, cudaMemcpyHostToDevice));
  C22_CUDA_CHECK(
      cudaMemcpy(d_ts_b, h_ts_b, ts_b_bytes, cudaMemcpyHostToDevice));

  // Compute features for both timeseries on GPU
  {
    int blk = C22_GPU_BLOCK_SIZE;
    c22_compute_features_kernel<<<(NA + blk - 1) / blk, blk>>>(
        d_ts_a, ts_a_length, window, NA, d_feat_a);
    C22_CUDA_CHECK(cudaGetLastError());
    c22_compute_features_kernel<<<(NB + blk - 1) / blk, blk>>>(
        d_ts_b, ts_b_length, window, NB, d_feat_b);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Initialise packed sentinel
  {
    unsigned long long s = sentinel_packed();
    std::vector<unsigned long long> init(NA, s);
    C22_CUDA_CHECK(
        cudaMemcpy(d_packed, init.data(), pack_bytes, cudaMemcpyHostToDevice));
  }

  // Tiled AB-join dot product
  {
    dim3 block(TILE_C, TILE_R);
    dim3 grid((NB + TILE_C - 1) / TILE_C, (NA + TILE_R - 1) / TILE_R);
    c22_tiled_abjoin_kernel<<<grid, block>>>(d_feat_a, d_feat_b, d_packed, NA,
                                             NB);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Decode packed → d_index
  {
    std::vector<unsigned long long> h_packed(NA);
    C22_CUDA_CHECK(cudaMemcpy(h_packed.data(), d_packed, pack_bytes,
                              cudaMemcpyDeviceToHost));
    std::vector<int> h_idx(NA);
    for (int i = 0; i < NA; ++i)
      h_idx[i] = (int)(unsigned int)(h_packed[i] & 0xFFFFFFFFULL);
    C22_CUDA_CHECK(
        cudaMemcpy(d_index, h_idx.data(), idx_bytes, cudaMemcpyHostToDevice));
    std::copy(h_idx.begin(), h_idx.end(), h_index);
  }

  // Exact double profile recompute
  {
    int thd = 256, blk = (NA + thd - 1) / thd;
    c22_recompute_abjoin_profile_kernel<<<blk, thd>>>(d_feat_a, d_feat_b,
                                                      d_index, d_profile, NA);
    C22_CUDA_CHECK(cudaGetLastError());
    C22_CUDA_CHECK(cudaDeviceSynchronize());
    C22_CUDA_CHECK(
        cudaMemcpy(h_profile, d_profile, prof_bytes, cudaMemcpyDeviceToHost));
  }

  C22_CUDA_CHECK(cudaFree(d_ts_a));
  C22_CUDA_CHECK(cudaFree(d_ts_b));
  C22_CUDA_CHECK(cudaFree(d_feat_a));
  C22_CUDA_CHECK(cudaFree(d_feat_b));
  C22_CUDA_CHECK(cudaFree(d_packed));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

}  // extern "C"

#endif  // _HAS_CUDA_
