#pragma once
#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void transpose(T *out, const T *in, int R, int C, int out_stride) {
  __shared__ T tile[32][33];

  int bx = blockIdx.x * 32;
  int by = blockIdx.y * 32;

  for (int i = 0; i < 32; i += 8) {
    int r = by + threadIdx.y + i;
    int c = bx + threadIdx.x;
    if (r < R && c < C)
      tile[threadIdx.y + i][threadIdx.x] = in[r * C + c];
  }
  __syncthreads();
  for (int i = 0; i < 32; i += 8) {
    int r = bx + threadIdx.y + i;
    int c = by + threadIdx.x;
    if (r < C && c < R)
      out[r * out_stride + c] = tile[threadIdx.x][threadIdx.y + i];
  }
}

template <typename T>
void launch_transpose_forward(T *out, const T *in, int R, int C,
                              int out_stride, cudaStream_t t) {
  dim3 block(32, 8);
  dim3 grid((C + 31) / 32, (R + 31) / 32);
  transpose<<<grid, block, 0, t>>>(out, in, R, C, out_stride);
}
