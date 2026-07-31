#include "../Gemm.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void init_bf16(__nv_bfloat16 *x, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		unsigned v = unsigned(i) * 747796405u + 2891336453u;
		v = ((v >> ((v >> 28) + 4)) ^ v) * 277803737u;
		v = (v >> 22) ^ v;
		x[i] = __float2bfloat16_rn(float(int(v & 0xffff) - 32768) / 65536.0f);
	}
}

template<typename F>
float time_us(F fn, int warmup, int iters) {
	for (int i = 0; i < warmup; ++i) fn();
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaEventRecord(begin);
	for (int i = 0; i < iters; ++i) fn();
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float ms;
	cudaEventElapsedTime(&ms, begin, end);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
	return ms * 1000.0f / iters;
}

int main() {
	constexpr int M = 1, N = 2048, K = 2048;
	constexpr int warmup = 200, iters = 2000, rounds = 7, copies = 16;
	using T = __nv_bfloat16;
	using Config = GemmConfig<T, T>;
	T *a, *b, *ours, *ref;
	cudaMalloc(&a, sizeof(T) * M * K);
	cudaMalloc(&b, sizeof(T) * K * N * copies);
	cudaMalloc(&ours, sizeof(T) * M * N);
	cudaMalloc(&ref, sizeof(T) * M * N);
	init_bf16<<<(M * K + 255) / 256, 256>>>(a, M * K);
	init_bf16<<<(K * N * copies + 255) / 256, 256>>>(b, K * N * copies);

	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1.0f, beta = 0.0f;
	int ours_index = 0, cublas_index = 0;
	auto run_ours = [&] {launch_Gemm_forward<Config>(a, b, ours, nullptr, alpha, beta, M, N, K, 0);};
	auto run_cublas = [&] {cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, CUDA_R_16BF, N, a, CUDA_R_16BF, K, &beta, ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);};
	auto run_ours_cold = [&] {launch_Gemm_forward<Config>(a, b + K * N * (ours_index++ % copies), ours, nullptr, alpha, beta, M, N, K, 0);};
	auto run_cublas_cold = [&] {cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b + K * N * (cublas_index++ % copies), CUDA_R_16BF, N, a, CUDA_R_16BF, K, &beta, ref, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);};
	std::vector<float> ours_times, cublas_times, ours_cold_times, cublas_cold_times;
	for (int i = 0; i < rounds; ++i) {
		if (i & 1) {
			cublas_times.push_back(time_us(run_cublas, warmup, iters));
			ours_times.push_back(time_us(run_ours, warmup, iters));
			cublas_cold_times.push_back(time_us(run_cublas_cold, warmup, iters));
			ours_cold_times.push_back(time_us(run_ours_cold, warmup, iters));
		} else {
			ours_times.push_back(time_us(run_ours, warmup, iters));
			cublas_times.push_back(time_us(run_cublas, warmup, iters));
			ours_cold_times.push_back(time_us(run_ours_cold, warmup, iters));
			cublas_cold_times.push_back(time_us(run_cublas_cold, warmup, iters));
		}
	}
	std::sort(ours_times.begin(), ours_times.end());
	std::sort(cublas_times.begin(), cublas_times.end());
	std::sort(ours_cold_times.begin(), ours_cold_times.end());
	std::sort(cublas_cold_times.begin(), cublas_cold_times.end());
	float ours_us = ours_times[rounds / 2], cublas_us = cublas_times[rounds / 2];
	float ours_cold_us = ours_cold_times[rounds / 2], cublas_cold_us = cublas_cold_times[rounds / 2];
	run_ours();
	run_cublas();
	cudaDeviceSynchronize();
	std::vector<T> h_ours(M * N), h_ref(M * N);
	cudaMemcpy(h_ours.data(), ours, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ref.data(), ref, sizeof(T) * M * N, cudaMemcpyDeviceToHost);
	float max_error = 0.0f;
	for (int i = 0; i < M * N; ++i) max_error = fmaxf(max_error, fabsf(float(h_ours[i]) - float(h_ref[i])));
	printf("M=%d N=%d K=%d hot: ours=%.3f us cublas=%.3f us speedup=%.3fx cold: ours=%.3f us cublas=%.3f us speedup=%.3fx max_error=%.6f\n", M, N, K, ours_us, cublas_us, cublas_us / ours_us, ours_cold_us, cublas_cold_us, cublas_cold_us / ours_cold_us, max_error);
	cublasDestroy(handle);
	cudaFree(a);
	cudaFree(b);
	cudaFree(ours);
	cudaFree(ref);
}
