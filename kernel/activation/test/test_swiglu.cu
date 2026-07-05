#include "../SwiGLU.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cmath>

// 标量版 baseline：每线程 1 个元素，无向量化
template<typename T>
__global__ void silu_mul_scalar(T *out, const T *gate, const T *up, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	float a = static_cast<float>(gate[idx]);
	float s = a / (1.0f + expf(-a));
	out[idx] = static_cast<T>(s * static_cast<float>(up[idx]));
}

// CPU 参照
template<typename T>
void silu_mul_cpu(T *out, const T *gate, const T *up, int N) {
	for (int i = 0; i < N; ++i) {
		float a = static_cast<float>(gate[i]);
		float s = a / (1.0f + expf(-a));
		out[i] = static_cast<T>(s * static_cast<float>(up[i]));
	}
}

struct TestSize { const char *label; int N; };

static const TestSize sizes[] = {
	{ "8960",   8960      },  // Qwen FFN 中间维度
	{ "tail",   8961      },  // 非向量对齐，覆盖标量尾巴分支
	{ "64K",    65536     },
	{ "1M",     1048576   },
	{ "16M",    16777216  },
	{ "128M",   134217728 },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== SwiGLU (silu_mul) %s =====\n", type_name);
	printf("%-8s %10s  %10s %10s  %8s %8s  %s\n",
			"size", "N", "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "accuracy");

	for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); ++si) {
		int N = sizes[si].N;
		size_t bytes = sizeof(T) * N;

		T *h_gate   = (T*)malloc(bytes);
		T *h_up     = (T*)malloc(bytes);
		T *h_ref    = (T*)malloc(bytes);
		T *h_scalar = (T*)malloc(bytes);
		T *h_vec    = (T*)malloc(bytes);
		fill_matrix(h_gate, N, -3.0f, 3.0f);
		fill_matrix(h_up, N, -3.0f, 3.0f);
		silu_mul_cpu(h_ref, h_gate, h_up, N);

		T *d_gate, *d_up, *d_out;
		CUDA_CHECK(cudaMalloc(&d_gate, bytes));
		CUDA_CHECK(cudaMalloc(&d_up, bytes));
		CUDA_CHECK(cudaMalloc(&d_out, bytes));
		CUDA_CHECK(cudaMemcpy(d_gate, h_gate, bytes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_up, h_up, bytes, cudaMemcpyHostToDevice));

		// 标量版
		int threads = 256;
		int blocks_s = (N + threads - 1) / threads;
		CUDA_CHECK(cudaMemset(d_out, 0, bytes));
		float ms_scalar = 0;
		TIME_MS(ms_scalar, 10, 100,
				silu_mul_scalar<<<blocks_s, threads>>>(d_out, d_gate, d_up, N));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_scalar, d_out, bytes, cudaMemcpyDeviceToHost));

		// 向量化版
		CUDA_CHECK(cudaMemset(d_out, 0, bytes));
		float ms_vec = 0;
		TIME_MS(ms_vec, 10, 100,
				launch_silu_mul(d_out, d_gate, d_up, N));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_vec, d_out, bytes, cudaMemcpyDeviceToHost));

		// 读 gate + 读 up + 写 out = 3N * sizeof(T) 字节
		double total_bytes = 3.0 * N * sizeof(T);
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

		cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
		free(h_gate); free(h_up); free(h_ref); free(h_scalar); free(h_vec);
	}
}

int main() {
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
	test_type<float>("f32");
}
