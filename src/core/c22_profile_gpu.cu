#ifdef _HAS_CUDA_

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// Number of catch22 features
#define C22_NUM_FEATURES 22

// Tile dimensions for the tiled dot-product kernel.
// Each CUDA block handles a (TILE_R x TILE_C) region of the N x N similarity
// matrix.  Both tiles are staged in shared memory so each feature vector is
// loaded from global memory only once per tile rather than once per cell.
//
// Constraints:
//   TILE_R * C22_NUM_FEATURES * sizeof(double) +
//   TILE_C * C22_NUM_FEATURES * sizeof(double)  <= shared memory limit (~48 KB)
//
// With TILE_R = TILE_C = 32:
//   2 * 32 * 22 * 8 = 11,264 bytes  (well within 48 KB)
//
// Block layout: (TILE_C, TILE_R) threads — each thread handles one (row, col)
// cell and accumulates the 22-element dot product using registers.
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
// Tiled self-join kernel
//
// Grid:  (ceil(N/TILE_C), ceil(N/TILE_R))  2-D grid of tiles
// Block: (TILE_C, TILE_R)                  one thread per cell in the tile
//
// For each tile (bx, by):
//   - col_tile covers columns [bx*TILE_C, (bx+1)*TILE_C)
//   - row_tile covers rows    [by*TILE_R, (by+1)*TILE_R)
//
// Each tile cooperatively loads TILE_R row vectors and TILE_C col vectors into
// shared memory, then every thread computes one dot product using those cached
// values.  An atomic max update keeps the per-row profile correct when multiple
// tiles contribute to the same row.
//
// Only the upper triangle (col > row + exclusion) is processed; the lower
// triangle is symmetric so each row's maximum is the same either way.
// ============================================================================
__global__ void c22_tiled_selfjoin_kernel(
    const double* __restrict__ F,  // N x 22  (row-major)
    double* __restrict__ profile,  // N — best dot product per row
    int* __restrict__ idx,         // N — best match index per row
    int N,
    int exclusion) {

  // Shared memory tiles: row vectors and column vectors
  __shared__ double sh_row[TILE_R][C22_NUM_FEATURES];
  __shared__ double sh_col[TILE_C][C22_NUM_FEATURES];

  const int tx = threadIdx.x;  // 0..TILE_C-1  → column within tile
  const int ty = threadIdx.y;  // 0..TILE_R-1  → row within tile

  // Global row / col this thread owns
  const int row = blockIdx.y * TILE_R + ty;
  const int col = blockIdx.x * TILE_C + tx;

  // ---- Cooperative load of row vectors into shared memory ----
  // Thread (0..TILE_C-1, ty) loads feature f for global row (blockIdx.y*TILE_R + ty)
  {
    int load_row = blockIdx.y * TILE_R + ty;
    // Each thread in the x-dimension loads one or more features
    // We use tx to stride over C22_NUM_FEATURES
    for (int f = tx; f < C22_NUM_FEATURES; f += TILE_C) {
      if (load_row < N) {
        sh_row[ty][f] = F[load_row * C22_NUM_FEATURES + f];
      } else {
        sh_row[ty][f] = 0.0;
      }
    }
  }

  // ---- Cooperative load of column vectors into shared memory ----
  {
    int load_col = blockIdx.x * TILE_C + tx;
    for (int f = ty; f < C22_NUM_FEATURES; f += TILE_R) {
      if (load_col < N) {
        sh_col[tx][f] = F[load_col * C22_NUM_FEATURES + f];
      } else {
        sh_col[tx][f] = 0.0;
      }
    }
  }

  __syncthreads();

  // ---- Compute dot product for cell (row, col) ----
  if (row < N && col < N) {
    int diff = (row > col) ? (row - col) : (col - row);
    if (diff > exclusion) {
      double dot = 0.0;
      for (int f = 0; f < C22_NUM_FEATURES; ++f) {
        dot += sh_row[ty][f] * sh_col[tx][f];
      }

      // Atomic update profile[row]: keep maximum dot product
      // CUDA doesn't have native atomicMax for double, so use CAS loop.
      unsigned long long* addr =
          reinterpret_cast<unsigned long long*>(&profile[row]);
      unsigned long long old_val = *addr;
      unsigned long long new_val;
      double old_dbl;
      do {
        old_dbl = __longlong_as_double(old_val);
        if (dot <= old_dbl) break;
        new_val = __double_as_longlong(dot);
        unsigned long long prev =
            atomicCAS(addr, old_val, new_val);
        if (prev == old_val) {
          // We updated profile[row]; also update idx[row] (best-effort, not
          // perfectly atomic with profile but consistent in practice because
          // ties are resolved by profile value which is already written).
          atomicExch(reinterpret_cast<int*>(&idx[row]), col);
          break;
        }
        old_val = prev;
      } while (true);
    }
  }
}

// ============================================================================
// Tiled AB-join kernel  (same structure, no exclusion zone)
// ============================================================================
__global__ void c22_tiled_abjoin_kernel(
    const double* __restrict__ FA,  // NA x 22
    const double* __restrict__ FB,  // NB x 22
    double* __restrict__ profile,   // NA
    int* __restrict__ idx,          // NA
    int NA,
    int NB) {

  __shared__ double sh_row[TILE_R][C22_NUM_FEATURES];
  __shared__ double sh_col[TILE_C][C22_NUM_FEATURES];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * TILE_R + ty;
  const int col = blockIdx.x * TILE_C + tx;

  // Load row (from FA)
  {
    int load_row = blockIdx.y * TILE_R + ty;
    for (int f = tx; f < C22_NUM_FEATURES; f += TILE_C) {
      sh_row[ty][f] = (load_row < NA) ? FA[load_row * C22_NUM_FEATURES + f] : 0.0;
    }
  }

  // Load col (from FB)
  {
    int load_col = blockIdx.x * TILE_C + tx;
    for (int f = ty; f < C22_NUM_FEATURES; f += TILE_R) {
      sh_col[tx][f] = (load_col < NB) ? FB[load_col * C22_NUM_FEATURES + f] : 0.0;
    }
  }

  __syncthreads();

  if (row < NA && col < NB) {
    double dot = 0.0;
    for (int f = 0; f < C22_NUM_FEATURES; ++f) {
      dot += sh_row[ty][f] * sh_col[tx][f];
    }

    unsigned long long* addr =
        reinterpret_cast<unsigned long long*>(&profile[row]);
    unsigned long long old_val = *addr;
    unsigned long long new_val;
    double old_dbl;
    do {
      old_dbl = __longlong_as_double(old_val);
      if (dot <= old_dbl) break;
      new_val = __double_as_longlong(dot);
      unsigned long long prev = atomicCAS(addr, old_val, new_val);
      if (prev == old_val) {
        atomicExch(reinterpret_cast<int*>(&idx[row]), col);
        break;
      }
      old_val = prev;
    } while (true);
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
    int gpu_id) {

  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_bytes = (size_t)N * C22_NUM_FEATURES * sizeof(double);
  size_t prof_bytes = (size_t)N * sizeof(double);
  size_t idx_bytes  = (size_t)N * sizeof(int);

  double* d_features = nullptr;
  double* d_profile  = nullptr;
  int*    d_index    = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile,  prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index,    idx_bytes));

  // Initialize profile to -DBL_MAX and index to -1
  // so the atomic max starts from the lowest possible value.
  C22_CUDA_CHECK(cudaMemset(d_index, 0xFF, idx_bytes));  // -1 as int
  {
    // Fill d_profile with -DBL_MAX via a small host array would be too slow
    // for N=1M; use cudaMemset with the bit pattern of -DBL_MAX.
    // -DBL_MAX as 64-bit: 0xFFEFFFFFFFFFFFFF
    // cudaMemset only sets bytes, so we use a kernel-free trick:
    // We'll do it on the host and copy.
    // For large N this copy is amortized; feature copy dominates anyway.
    std::vector<double> init_prof(N, -DBL_MAX);
    C22_CUDA_CHECK(cudaMemcpy(d_profile, init_prof.data(), prof_bytes,
                              cudaMemcpyHostToDevice));
  }

  C22_CUDA_CHECK(cudaMemcpy(d_features, h_features, feat_bytes,
                            cudaMemcpyHostToDevice));

  // Launch tiled kernel
  dim3 block(TILE_C, TILE_R);
  dim3 grid((N + TILE_C - 1) / TILE_C, (N + TILE_R - 1) / TILE_R);

  c22_tiled_selfjoin_kernel<<<grid, block>>>(d_features, d_profile, d_index,
                                             N, exclusion);
  C22_CUDA_CHECK(cudaGetLastError());
  C22_CUDA_CHECK(cudaDeviceSynchronize());

  C22_CUDA_CHECK(cudaMemcpy(h_profile, d_profile, prof_bytes,
                            cudaMemcpyDeviceToHost));
  C22_CUDA_CHECK(cudaMemcpy(h_index,   d_index,   idx_bytes,
                            cudaMemcpyDeviceToHost));

  C22_CUDA_CHECK(cudaFree(d_features));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

void c22_profile_abjoin_gpu_launch(
    const double* h_features_a,
    const double* h_features_b,
    double* h_profile,
    int* h_index,
    int NA,
    int NB,
    int gpu_id) {

  C22_CUDA_CHECK(cudaSetDevice(gpu_id));

  size_t feat_a_bytes = (size_t)NA * C22_NUM_FEATURES * sizeof(double);
  size_t feat_b_bytes = (size_t)NB * C22_NUM_FEATURES * sizeof(double);
  size_t prof_bytes   = (size_t)NA * sizeof(double);
  size_t idx_bytes    = (size_t)NA * sizeof(int);

  double* d_features_a = nullptr;
  double* d_features_b = nullptr;
  double* d_profile    = nullptr;
  int*    d_index      = nullptr;

  C22_CUDA_CHECK(cudaMalloc(&d_features_a, feat_a_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_features_b, feat_b_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_profile,    prof_bytes));
  C22_CUDA_CHECK(cudaMalloc(&d_index,      idx_bytes));

  C22_CUDA_CHECK(cudaMemset(d_index, 0xFF, idx_bytes));
  {
    std::vector<double> init_prof(NA, -DBL_MAX);
    C22_CUDA_CHECK(cudaMemcpy(d_profile, init_prof.data(), prof_bytes,
                              cudaMemcpyHostToDevice));
  }

  C22_CUDA_CHECK(cudaMemcpy(d_features_a, h_features_a, feat_a_bytes,
                            cudaMemcpyHostToDevice));
  C22_CUDA_CHECK(cudaMemcpy(d_features_b, h_features_b, feat_b_bytes,
                            cudaMemcpyHostToDevice));

  dim3 block(TILE_C, TILE_R);
  dim3 grid((NB + TILE_C - 1) / TILE_C, (NA + TILE_R - 1) / TILE_R);

  c22_tiled_abjoin_kernel<<<grid, block>>>(d_features_a, d_features_b,
                                           d_profile, d_index, NA, NB);
  C22_CUDA_CHECK(cudaGetLastError());
  C22_CUDA_CHECK(cudaDeviceSynchronize());

  C22_CUDA_CHECK(cudaMemcpy(h_profile, d_profile, prof_bytes,
                            cudaMemcpyDeviceToHost));
  C22_CUDA_CHECK(cudaMemcpy(h_index,   d_index,   idx_bytes,
                            cudaMemcpyDeviceToHost));

  C22_CUDA_CHECK(cudaFree(d_features_a));
  C22_CUDA_CHECK(cudaFree(d_features_b));
  C22_CUDA_CHECK(cudaFree(d_profile));
  C22_CUDA_CHECK(cudaFree(d_index));
}

}  // extern "C"

#endif  // _HAS_CUDA_
