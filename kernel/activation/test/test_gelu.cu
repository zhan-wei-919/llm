#include "../GELU.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cmath>

// 标量版 baseline：每线程 1 个元素，无向量化
template<typename T>
__global__ void GELU_scalar(T *y, const T *x, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	float xc = static_cast<float>(x[idx]);
	float inner = 0.7978845608f * (xc + 0.044715f * xc*xc*xc);
	y[idx] = static_cast<T>(0.5f * xc * (1.0f + tanhf(inner)));
}

// CPU 参照
template<typename T>
void gelu_cpu(T *y, const T *x, int N) {
	for (int i = 0; i < N; ++i) {
		float xc = static_cast<float>(x[i]);
		float inner = 0.7978845608f * (xc + 0.044715f * xc*xc*xc);
		y[i] = static_cast<T>(0.5f * xc * (1.0f + tanhf(inner)));
	}
}

struct TestSize { const char *label; int N; };

static const TestSize sizes[] = {
	{ "4K",     4096         },
	{ "64K",    65536        },
	{ "1M",     1048576      },
	{ "16M",    16777216     },
	{ "64M",    67108864     },
	{ "128M",   134217728    },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== GELU %s =====\n", type_name);
	printf("%-8s %10s  %10s %10s  %8s %8s  %s\n",
			"size", "N", "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "accuracy");

	double peak_bw = 0;

	for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); ++si) {
		int N = sizes[si].N;
		size_t bytes = sizeof(T) * N;

		T *h_x = (T*)malloc(bytes);
		T *h_ref = (T*)malloc(bytes);
		T *h_scalar = (T*)malloc(bytes);
		T *h_vec = (T*)malloc(bytes);
		fill_matrix(h_x, N, -3.0f, 3.0f);
		gelu_cpu(h_ref, h_x, N);

		T *d_x, *d_y;
		CUDA_CHECK(cudaMalloc(&d_x, bytes));
		CUDA_CHECK(cudaMalloc(&d_y, bytes));
		CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

		// 标量版
		int threads = 256;
		int blocks_s = (N + threads - 1) / threads;
		CUDA_CHECK(cudaMemset(d_y, 0, bytes));
		float ms_scalar = 0;
		TIME_MS(ms_scalar, 10, 100,
				GELU_scalar<<<blocks_s, threads>>>(d_y, d_x, N));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_scalar, d_y, bytes, cudaMemcpyDeviceToHost));

		// 向量化版
		CUDA_CHECK(cudaMemset(d_y, 0, bytes));
		float ms_vec = 0;
		TIME_MS(ms_vec, 10, 100,
				launch_GELU_forward(d_y, d_x, N));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_vec, d_y, bytes, cudaMemcpyDeviceToHost));

		// GELU 读 N 个 T + 写 N 个 T = 2N * sizeof(T) 字节
		double total_bytes = 2.0 * N * sizeof(T);
		double bw_vec = total_bytes / (ms_vec * 1e-3) / 1e9;

		bool ok_scalar = matricesEqual(h_scalar, h_ref, N, 1e-3f);
		bool ok_vec    = matricesEqual(h_vec, h_ref, N, 1e-3f);

		float speedup = ms_scalar / ms_vec;

		printf("%-8s %10d  %10.1f %10.1f  %7.2fx %7.1f  %s/%s\n",
				sizes[si].label, N,
				ms_scalar * 1000, ms_vec * 1000,
				speedup, bw_vec,
				ok_scalar ? "PASS" : "FAIL",
				ok_vec ? "PASS" : "FAIL");

		cudaFree(d_x); cudaFree(d_y);
		free(h_x); free(h_ref); free(h_scalar); free(h_vec);
	}

	printf("(peak DRAM BW: %.1f GB/s)\n", peak_bw);
}

int main() {
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
	test_type<float>("f32");
}
