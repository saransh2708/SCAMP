#include "scamp_interface.h"

#include <chrono>
#include <thread>
#include <vector>

#include "core/SCAMP.h"
#include "core/c22_profile.h"
#include "scamp_exception.h"

namespace SCAMP {

int num_available_gpus() {
  int num_dev = 0;
#ifdef _HAS_CUDA_
  cudaGetDeviceCount(&num_dev);
#endif
  return num_dev;
}

void do_SCAMP(SCAMPArgs *args) {
  int num_threads = 0;
  int num_devices = num_available_gpus();
  std::vector<int> devices(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    devices.at(i) = i;
  }
  if (devices.empty()) {
    num_threads = std::thread::hardware_concurrency();
  }
  do_SCAMP(args, devices, num_threads);
}

// ============================================================================
// C22 Profile: replaces MASS with C22 feature vectors + dot product.
//
// This is dispatched from do_SCAMP() when profile_type == PROFILE_TYPE_C22.
// Feature vectors are computed in parallel on CPU, and the O(N^2) dot product
// search is executed on GPU (if available) or CPU.  The result is packed into
// the Profile structures so that the caller (Python bindings, CLI, distributed
// worker) can extract results in the standard way.
// ============================================================================
static void do_SCAMP_C22(SCAMPArgs *args, const std::vector<int> &devices,
                         int num_threads) {
  if (!args->silent_mode) {
    std::cout << "C22 Profile: replacing MASS with C22 feature vectors + dot "
                 "product."
              << std::endl;
  }

  // Determine CPU thread count (for parallel C22 feature computation)
  int cpu_threads = num_threads;
  if (cpu_threads <= 0) {
    cpu_threads = std::thread::hardware_concurrency();
    if (cpu_threads <= 0) cpu_threads = 1;
  }

  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();

  C22ProfileResult result;

  if (!args->has_b) {
    // Self-join
#ifdef _HAS_CUDA_
    if (!devices.empty()) {
      if (!args->silent_mode) {
        std::cout << "C22 self-join: GPU " << devices[0]
                  << " (feature computation on " << cpu_threads
                  << " CPU threads)" << std::endl;
      }
      result = c22_profile_selfjoin_gpu(args->timeseries_a, args->window,
                                        devices[0], cpu_threads);
    } else
#endif
    {
      if (!args->silent_mode) {
        std::cout << "C22 self-join: " << cpu_threads << " CPU threads"
                  << std::endl;
      }
      result = c22_profile_selfjoin_cpu(args->timeseries_a, args->window,
                                        cpu_threads);
    }
  } else {
    // AB-join
#ifdef _HAS_CUDA_
    if (!devices.empty()) {
      if (!args->silent_mode) {
        std::cout << "C22 AB-join: GPU " << devices[0]
                  << " (feature computation on " << cpu_threads
                  << " CPU threads)" << std::endl;
      }
      result = c22_profile_abjoin_gpu(args->timeseries_a, args->timeseries_b,
                                      args->window, devices[0], cpu_threads);
    } else
#endif
    {
      if (!args->silent_mode) {
        std::cout << "C22 AB-join: " << cpu_threads << " CPU threads"
                  << std::endl;
      }
      result = c22_profile_abjoin_cpu(args->timeseries_a, args->timeseries_b,
                                      args->window, cpu_threads);
    }
  }

  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();

  if (!args->silent_mode) {
    double elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        static_cast<double>(1000000);
    std::cout << "C22 Profile computed in " << elapsed << " seconds."
              << std::endl;
  }

  // Pack results into Profile structure
  args->profile_a.type = PROFILE_TYPE_C22;
  args->profile_a.data.clear();
  args->profile_a.data.emplace_back();
  args->profile_a.data[0].double_value = std::move(result.profile);
  args->profile_a.data[0].uint64_value.resize(result.index.size());
  for (size_t i = 0; i < result.index.size(); ++i) {
    args->profile_a.data[0].uint64_value[i] =
        static_cast<uint64_t>(result.index[i]);
  }
}

// Wrapper on SCAMP_Operation called by main(), this function constructs
// and executes a SCAMP_Operation given a set of user selected parameters.
void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads) {
  if (devices.empty() && num_threads <= 0) {
    throw SCAMPException("Error: no compute_resources provided");
  }

  if (args == nullptr) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  if (!args->silent_mode) {
    std::cout << "Validating SCAMP args." << std::endl;
  }
  args->validate();

  // ---- C22 Profile: replaces MASS with C22 feature vectors + dot product ----
  // Dispatched here instead of through SCAMP_Operation because the C22
  // similarity metric (dot product of 22-element feature vectors) does not
  // benefit from SCAMP's diagonal traversal / incremental QT update.
  if (args->profile_type == PROFILE_TYPE_C22) {
    do_SCAMP_C22(args, devices, num_threads);
    return;
  }

  // Allocate and initialize memory
  if (!args->InitProfileMemory()) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  OptionalArgs _opt_args(args->distance_threshold);

  if (!args->silent_mode) {
    std::cout << "Building SCAMP Operation from args" << std::endl;
  }

  // Construct operation
  SCAMP_Operation op(
      args->timeseries_a.size(), args->timeseries_b.size(), args->window,
      args->max_tile_size, devices, !args->has_b, args->precision_type,
      args->distributed_start_row, args->distributed_start_col, _opt_args,
      args->profile_type, &args->profile_a, &args->profile_b,
      args->keep_rows_separate, args->computing_rows, args->computing_columns,
      args->is_aligned, args->silent_mode, num_threads,
      args->max_matches_per_column, args->matrix_height, args->matrix_width);

  if (!args->silent_mode) {
    std::cout << "SCAMP Operation constructed" << std::endl;
  }
  // Execute op
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  if (args->has_b) {
    op.do_join(args->timeseries_a, args->timeseries_b);
  } else {
    op.do_join(args->timeseries_a, args->timeseries_a);
  }
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  if (!args->silent_mode) {
    printf(
        "Finished %d SCAMP tiles to generate  matrix profile in %lf "
        "seconds on %lu devices and %d threads\n",
        op.get_completed_tiles(),
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            static_cast<double>(1000000),
        devices.size(), num_threads);
  }
}

}  // namespace SCAMP
