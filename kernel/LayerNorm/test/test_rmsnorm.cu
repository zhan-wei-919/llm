#include "../RMSNorm.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 标量参考 kernel：每个 block 处理一行 [C]，不做向量化，用来和向量化实现交叉验证
template<typename T>
__global__ void rmsnorm_scalar(
		T *out,
		const T *x, const T *gamma,
		int C, float eps
) {
	int bt = blockIdx.x;
	float sum2 = 0.0f;
	for (int c = threadIdx.x; c < C; c += blockDim.x) {
		float v = static_cast<float>(x[bt * C + c]);
		sum2 += v * v;
	}
	sum2 = block_sum(sum2);
	float rstd = rsqrtf(sum2 / C + eps);
	for (int c = threadIdx.x; c < C; c += blockDim.x) {
		float v = static_cast<float>(x[bt * C + c]);
		float g = static_cast<float>(gamma[c]);
		out[bt * C + c] = static_cast<T>(v * rstd * g);
	}
}

// CPU 参考实现，全程 float 累加，作为精度基准
template<typename T>
void rmsnorm_cpu(
		float *out,
		const T *x, const T *gamma,
		int BT, int C, float eps
) {
	for (int bt = 0; bt < BT; ++bt) {
		float sum2 = 0.0f;
		for (int c = 0; c < C; ++c) {
			float v = static_cast<float>(x[bt * C + c]);
			sum2 += v * v;
		}
		float rstd = 1.0f / sqrtf(sum2 / C + eps);
		for (int c = 0; c < C; ++c) {
			float v = static_cast<float>(x[bt * C + c]);
			float g = static_cast<float>(gamma[c]);
			out[bt * C + c] = v * rstd * g;
		}
	}
}

struct TestCase { const char *label; int BT, C; };

static const TestCase cases[] = {
	{ "small",    128,   768  },
	{ "gpt2",   4096,   768  },
	{ "gpt2-lg", 4096,  1280 },
	{ "llama",   4096,  4096 },
	{ "big",     8192,  4096 },
};

static constexpr int BLOCK = 256;
static constexpr float EPS = 1e-5f;

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== RMSNorm %s =====\n", type_name);
	printf("  %-8s %6s %5s  %10s %10s  %7s %9s  %s\n",
			"case", "BT", "C", "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "acc");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int BT = cases[ci].BT, C = cases[ci].C;
		size_t s_x     = sizeof(T) * BT * C;
		size_t s_out   = sizeof(T) * BT * C;
		size_t s_param = sizeof(T) * C;

		T     *h_x        = (T*)malloc(s_x);
		T     *h_gamma    = (T*)malloc(s_param);
		float *h_ref_out  = (float*)malloc(sizeof(float) * BT * C);
		T     *h_scalar   = (T*)malloc(s_out);
		T     *h_vec      = (T*)malloc(s_out);

		fill_matrix(h_x, BT * C, -1.0f, 1.0f);
		fill_matrix(h_gamma, C, 0.5f, 1.5f);

		rmsnorm_cpu(h_ref_out, h_x, h_gamma, BT, C, EPS);

		T *d_x, *d_out, *d_gamma;
		CUDA_CHECK(cudaMalloc(&d_x, s_x));
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMalloc(&d_gamma, s_param));
		CUDA_CHECK(cudaMemcpy(d_x, h_x, s_x, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, s_param, cudaMemcpyHostToDevice));

		// scalar
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_scalar = 0;
		TIME_MS(ms_scalar, 20, 200,
				(rmsnorm_scalar<T><<<BT, BLOCK>>>(
					d_out, d_x, d_gamma, C, EPS)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_scalar, d_out, s_out, cudaMemcpyDeviceToHost));

		// vectorized
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_vec = 0;
		TIME_MS(ms_vec, 20, 200,
				(rmsnorm<T><<<BT, BLOCK>>>(
					d_out, d_x, d_gamma, C, EPS)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_vec, d_out, s_out, cudaMemcpyDeviceToHost));

		// 读x + 读gamma + 写out = (2*BT*C + C) * sizeof(T)，C很小时gamma可忽略
		double total_bytes = (2.0 * BT * C + 1.0 * C) * sizeof(T);
		double bw = total_bytes / (ms_vec * 1e-3) / 1e9;

		bool ok_s = matricesEqual(h_scalar, h_ref_out, BT * C, 2e-2f);
		bool ok_v = matricesEqual(h_vec, h_ref_out, BT * C, 2e-2f);
		float speedup = ms_scalar / ms_vec;

		printf("  %-8s %6d %5d  %10.1f %10.1f  %6.2fx %8.1f  %s/%s\n",
				cases[ci].label, BT, C,
				ms_scalar * 1000, ms_vec * 1000,
				speedup, bw,
				ok_s ? "PASS" : "FAIL",
				ok_v ? "PASS" : "FAIL");

		cudaFree(d_x); cudaFree(d_out); cudaFree(d_gamma);
		free(h_x); free(h_gamma);
		free(h_ref_out); free(h_scalar); free(h_vec);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
