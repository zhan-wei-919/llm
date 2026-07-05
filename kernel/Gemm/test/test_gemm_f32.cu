#include "../Gemm_f32.cuh"
#include "../../utils.cuh"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

struct TestCase { const char *label; int M, N, K; };

static const TestCase cases[] = {
	{ "tiny",      64,    64,    64 },
	{ "small",    128,   128,   128 },
	{ "medium",   256,   256,   256 },
	{ "gpt2-qkv", 128,  2304,   768 },
	{ "gpt2-ffn", 128,  3072,   768 },
	{ "square",   512,   512,   512 },
	{ "large",   1024,  1024,  1024 },
	{ "xlarge",  2048,  2048,  2048 },
	{ "4k",      4096,  4096,  4096 },
};

void test_gemm_f32() {
	printf("\n===== Gemm f32: ours vs cuBLAS =====\n");
	printf("  %-10s %5s %5s %5s  %10s %8s  %10s %8s  %7s  %s\n",
			"case", "M", "N", "K",
			"ours(us)", "GFLOPS", "cublas(us)", "GFLOPS", "ratio", "accuracy");

	cublasHandle_t handle;
	cublasCreate(&handle);

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int M = cases[ci].M, N = cases[ci].N, K = cases[ci].K;

		size_t s_a = sizeof(float) * M * K;
		size_t s_b = sizeof(float) * K * N;
		size_t s_c = sizeof(float) * M * N;
		size_t s_bias = sizeof(float) * N;

		float *h_a      = (float*)malloc(s_a);
		float *h_b      = (float*)malloc(s_b);
		float *h_bias   = (float*)malloc(s_bias);
		float *h_ours   = (float*)malloc(s_c);
		float *h_cublas = (float*)malloc(s_c);

		fill_matrix(h_a, M * K, -0.5f, 0.5f);
		fill_matrix(h_b, K * N, -0.5f, 0.5f);
		fill_matrix(h_bias, N, -0.1f, 0.1f);

		float alpha = 1.0f, beta = 0.0f;

		float *d_a, *d_b, *d_c, *d_bias;
		CUDA_CHECK(cudaMalloc(&d_a, s_a));
		CUDA_CHECK(cudaMalloc(&d_b, s_b));
		CUDA_CHECK(cudaMalloc(&d_c, s_c));
		CUDA_CHECK(cudaMalloc(&d_bias, s_bias));
		CUDA_CHECK(cudaMemcpy(d_a, h_a, s_a, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_b, h_b, s_b, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_bias, h_bias, s_bias, cudaMemcpyHostToDevice));

		// ---- ours ----
		CUDA_CHECK(cudaMemset(d_c, 0, s_c));
		float ms_ours = 0;
		TIME_MS(ms_ours, 5, 50,
				launch_gemm_f32_forward(d_a, d_b, d_c, d_bias, alpha, beta, M, N, K));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_ours, d_c, s_c, cudaMemcpyDeviceToHost));

		// ---- cuBLAS ----
		// cuBLAS 是列主序，对行主序的 C = A * B，等价于 C^T = B^T * A^T
		// cublasSgemm(handle, transB, transA, N, M, K, &alpha, B, N, A, K, &beta, C, N)
		CUDA_CHECK(cudaMemset(d_c, 0, s_c));
		float ms_cublas = 0;
		float cublas_beta = 0.0f;
		TIME_MS(ms_cublas, 5, 50,
				cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
						N, M, K, &alpha, d_b, N, d_a, K, &cublas_beta, d_c, N));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_cublas, d_c, s_c, cudaMemcpyDeviceToHost));

		// cuBLAS 结果不含 bias，手动加上后比较
		for (int m = 0; m < M; ++m)
			for (int n = 0; n < N; ++n)
				h_cublas[m * N + n] += h_bias[n];

		double flops = 2.0 * M * N * K;
		double gflops_ours   = flops / (ms_ours * 1e-3) / 1e9;
		double gflops_cublas = flops / (ms_cublas * 1e-3) / 1e9;
		float ratio = ms_cublas / ms_ours;

		float tol = 1e-2f;
		bool ok = matricesEqual(h_ours, h_cublas, M * N, tol);

		printf("  %-10s %5d %5d %5d  %10.1f %8.1f  %10.1f %8.1f  %6.2fx  %s\n",
				cases[ci].label, M, N, K,
				ms_ours * 1000, gflops_ours,
				ms_cublas * 1000, gflops_cublas,
				ratio,
				ok ? "PASS" : "FAIL");

		if (!ok) {
			float max_err = 0;
			for (int i = 0; i < M * N; ++i) {
				float diff = fabsf(h_ours[i] - h_cublas[i]);
				if (diff > max_err) max_err = diff;
			}
			printf("           max_err=%.6f\n", max_err);
		}

		cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_bias);
		free(h_a); free(h_b); free(h_bias);
		free(h_ours); free(h_cublas);
	}

	cublasDestroy(handle);
}

int main() {
	test_gemm_f32();
}
