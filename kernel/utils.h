#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _cuda_check_err = (call);                                      \
    if (_cuda_check_err != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n", __FILE__, __LINE__,    \
              #call, cudaGetErrorString(_cuda_check_err));                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define TIME_MS(out_ms, warmup, iters, ...)                                    \
  do {                                                                         \
    cudaEvent_t _time_ms_start_t, _time_ms_end_t;                              \
    CUDA_CHECK(cudaEventCreate(&_time_ms_start_t));                            \
    CUDA_CHECK(cudaEventCreate(&_time_ms_end_t));                              \
    for (int _time_ms_i = 0; _time_ms_i < (warmup); ++_time_ms_i) {            \
      __VA_ARGS__;                                                             \
    }                                                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
    CUDA_CHECK(cudaEventRecord(_time_ms_start_t));                             \
    for (int _time_ms_i = 0; _time_ms_i < (iters); ++_time_ms_i) {             \
      __VA_ARGS__;                                                             \
    }                                                                          \
    CUDA_CHECK(cudaEventRecord(_time_ms_end_t));                               \
    CUDA_CHECK(cudaEventSynchronize(_time_ms_end_t));                          \
    float _time_ms = 0.0f;                                                     \
    CUDA_CHECK(                                                                \
        cudaEventElapsedTime(&_time_ms, _time_ms_start_t, _time_ms_end_t));    \
    (out_ms) = _time_ms / (iters);                                             \
    CUDA_CHECK(cudaEventDestroy(_time_ms_start_t));                            \
    CUDA_CHECK(cudaEventDestroy(_time_ms_end_t));                              \
  } while (0)

template <typename T>
inline void fill_matrix(T *A, int size, float range_a, float range_b) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(range_a, range_b);
  for (int i = 0; i < size; i++) {
    A[i] = static_cast<T>(dist(rng));
  }
}

template <typename A, typename B>
inline bool matricesEqual(const A *a, const B *b, int size, float tolerance) {
  for (int i = 0; i < size; ++i) {
    float fa = static_cast<float>(a[i]);
    float fb = static_cast<float>(b[i]);
    float denom = max(abs(fa), abs(fb));
    if (abs(fa - fb) > tolerance * denom + 1e-3f) {
      return false;
    }
  }
  return true;
}
