#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ inline float          device_max(float a, float b)                    { return fmaxf(a, b); }
__device__ inline __nv_bfloat16  device_max(__nv_bfloat16 a, __nv_bfloat16 b)    { return __hmax(a, b); }
__device__ inline half           device_max(half a, half b)                      { return __hmax(a, b); }
__device__ inline float          device_max(float a, __nv_bfloat16 b)            { return fmaxf(a, static_cast<float>(b)); }
__device__ inline float          device_max(__nv_bfloat16 a, float b)            { return fmaxf(static_cast<float>(a), b); }
__device__ inline float          device_max(float a, half b)                     { return fmaxf(a, static_cast<float>(b)); }
__device__ inline float          device_max(half a, float b)                     { return fmaxf(static_cast<float>(a), b); }

template<typename T>
__device__ inline float warp_sum(T val) {
		float tmp = static_cast<float>(val);
		for (int delta = 16; delta >= 1; delta /= 2) {
				tmp += static_cast<float>(__shfl_down_sync(0xffffffff, tmp, delta));
		}
		return tmp;
}

template<typename T>
__device__ inline float block_sum(T val) {
		__shared__ float smem[32];
		int lane_id = threadIdx.x & 31;
		int warp_id = threadIdx.x >> 5;
		
		float tmp = warp_sum(val);
		if (lane_id == 0) smem[warp_id] = tmp;
		__syncthreads();
		
		int warp_nums = (blockDim.x + 31) / 32;
		tmp = (threadIdx.x < warp_nums)? smem[lane_id]:0.0f;
		if (warp_id == 0) tmp = warp_sum(tmp);
		
		if (threadIdx.x == 0) smem[0] = tmp;
		__syncthreads();
		float res = smem[0];
		__syncthreads();
		return res;
}

template<typename T>
__device__ inline T warp_max(T val) {
		for (int delta = 16; delta >= 1; delta /= 2) {
				val = device_max(val, __shfl_down_sync(0xffffffff, val, delta));
		}
		return val;
}

template<typename T>
__device__ inline T block_max(T val) {
		__shared__ T smem[32];
		int lane_id = threadIdx.x & 31;
		int warp_id = threadIdx.x >> 5;
		
		val = warp_max(val);
		if (lane_id == 0) smem[warp_id] = val;
		__syncthreads();
		
		int warp_nums = (blockDim.x + 31) / 32;
		val = (threadIdx.x < warp_nums)?smem[lane_id]:-INFINITY;
		if (warp_id == 0) val = warp_max(val);
		
		if (threadIdx.x == 0) smem[0] = val;
		__syncthreads();
		float res = smem[0];
		__syncthreads();
		return res;
}