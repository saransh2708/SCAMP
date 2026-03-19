include(SCAMPMacros)

# Build targets for computing Sliding Dot Product.
add_library(qt_helper ${CMAKE_CURRENT_SOURCE_DIR}/qt_helper.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(qt_kernels ${CMAKE_CURRENT_SOURCE_DIR}/qt_kernels.cu)
  target_link_libraries(qt_helper CUDA::cufft qt_kernels common)
else()
  target_link_libraries(qt_helper common)
endif()

# Build SCAMP GPU/CPU kernel targets.
add_library(kernel_common ${CMAKE_CURRENT_SOURCE_DIR}/kernel_common.cpp)

if (CMAKE_CUDA_COMPILER)
  add_library(gpu_utils ${CMAKE_CURRENT_SOURCE_DIR}/kernel_gpu_utils.cu)
  target_link_libraries(gpu_utils kernel_common)
  add_library(gpu_kernels ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cu)
  target_link_libraries(gpu_kernels gpu_utils kernel_common common)
endif()

add_library(cpu_stats ${CMAKE_CURRENT_SOURCE_DIR}/cpu_stats.cpp)
target_link_libraries(cpu_stats common)

# C22 Feature Vector Library
# Collect all C22 C source files from pycatch22 (CPU-compatible, no FFTW)
file(GLOB C22_C_SOURCES
  "${CMAKE_SOURCE_DIR}/pycatch22/src/C/*.c"
)
# Exclude main.c, test files, and Python wrapper
list(FILTER C22_C_SOURCES EXCLUDE REGEX ".*main\\.c$")
list(FILTER C22_C_SOURCES EXCLUDE REGEX ".*test\\.c$")
list(FILTER C22_C_SOURCES EXCLUDE REGEX ".*temp\\.c$")
list(FILTER C22_C_SOURCES EXCLUDE REGEX ".*catch22_wrap\\.c$")

# Create C22 features library (standalone — no dependency on SCAMP common)
add_library(c22_features
  ${CMAKE_CURRENT_SOURCE_DIR}/c22_features.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/c22_autocorr_opt.c
  ${C22_C_SOURCES}
)
target_include_directories(c22_features PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/pycatch22/src/C
)
# Set C standard for pycatch22 C files
set_source_files_properties(${C22_C_SOURCES} PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
)
# Apply same optimization flags to pycatch22 C files for SIMD auto-vectorization (Level 4)
set_source_files_properties(${C22_C_SOURCES} PROPERTIES
  COMPILE_FLAGS "${SCAMP_OPTIMIZED_BUILD_FLAGS}"
)
# c22_autocorr_opt.c uses pycatch22 headers — same C standard
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/c22_autocorr_opt.c PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
)

# C22 Profile computation engine
if (CMAKE_CUDA_COMPILER)
  # c22_profile_gpu_lib: tiled dot-product kernels (original)
  # c22_features_gpu_lib: per-subsequence feature extraction kernels (new)
  # Both are compiled into a single CUDA object library so they share device
  # symbols (e.g. c22_compute_features_kernel referenced from c22_profile_gpu).
  add_library(c22_profile_gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/c22_profile_gpu.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/c22_features_gpu.cu
  )
  target_compile_definitions(c22_profile_gpu PUBLIC -D_HAS_CUDA_)
  add_library(c22_profile ${CMAKE_CURRENT_SOURCE_DIR}/c22_profile.cpp)
  target_link_libraries(c22_profile c22_features c22_profile_gpu Threads::Threads)
  target_compile_definitions(c22_profile PUBLIC -D_HAS_CUDA_)
else()
  add_library(c22_profile ${CMAKE_CURRENT_SOURCE_DIR}/c22_profile.cpp)
  target_link_libraries(c22_profile c22_features Threads::Threads)
endif()

add_library(tile ${CMAKE_CURRENT_SOURCE_DIR}/tile.cpp)

# Defines 'cpu_kernels' target.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernel)

if (CMAKE_CUDA_COMPILER)
  target_link_libraries(tile common gpu_kernels cpu_kernels profile qt_helper)
else()
  target_link_libraries(tile common cpu_kernels profile qt_helper)
endif()

add_library(scamp_op ${CMAKE_CURRENT_SOURCE_DIR}/SCAMP.cpp)
target_link_libraries(scamp_op Threads::Threads tile cpu_stats common profile qt_helper)

mark_cuda_if_available()
mark_clang_tidy()
