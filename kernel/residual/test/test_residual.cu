#include "../Residual.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>

template<typename T>
__global__ void residual_scalar(T *out, const T *a, const T *b, int C) {
	int bt = blockIdx.x;
	for (int c = threadIdx.x; c < C; c += blockDim.x) {
		out[bt * C + c] = static_cast<T>(
				static_cast<float>(a[bt * C + c]) + static_cast<float>(b[bt * C + c]));
	}
}

template<typename T>
void residual_cpu(float *out, const T *a, const T *b, int BT, int C) {
	for (int bt = 0; bt < BT; ++bt)
		for (int c = 0; c < C; ++c)
			out[bt * C + c] = static_cast<float>(a[bt * C + c])
					+ static_cast<float>(b[bt * C + c]);
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

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== Residual %s =====\n", type_name);
	printf("  %-8s %6s %5s  %10s %10s  %7s %9s  %s\n",
			"case", "BT", "C", "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "acc");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int BT = cases[ci].BT, C = cases[ci].C;
		size_t s_data = sizeof(T) * BT * C;

		T     *h_a   = (T*)malloc(s_data);
		T     *h_b   = (T*)malloc(s_data);
		float *h_ref = (float*)malloc(sizeof(float) * BT * C);
		T     *h_scalar = (T*)malloc(s_data);
		T     *h_vec    = (T*)malloc(s_data);

		fill_matrix(h_a, BT * C, -1.0f, 1.0f);
		fill_matrix(h_b, BT * C, -1.0f, 1.0f);

		residual_cpu(h_ref, h_a, h_b, BT, C);

		T *d_a, *d_b, *d_out;
		CUDA_CHECK(cudaMalloc(&d_a, s_data));
		CUDA_CHECK(cudaMalloc(&d_b, s_data));
		CUDA_CHECK(cudaMalloc(&d_out, s_data));
		CUDA_CHECK(cudaMemcpy(d_a, h_a, s_data, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_b, h_b, s_data, cudaMemcpyHostToDevice));

		// scalar
		CUDA_CHECK(cudaMemset(d_out, 0, s_data));
		float ms_scalar = 0;
		TIME_MS(ms_scalar, 20, 200,
				(residual_scalar<T><<<BT, BLOCK>>>(d_out, d_a, d_b, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_scalar, d_out, s_data, cudaMemcpyDeviceToHost));

		// vectorized
		CUDA_CHECK(cudaMemset(d_out, 0, s_data));
		float ms_vec = 0;
		TIME_MS(ms_vec, 20, 200,
				(residual<T><<<BT, BLOCK>>>(d_out, d_a, d_b, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_vec, d_out, s_data, cudaMemcpyDeviceToHost));

		// 读a + 读b + 写out = 3 * BT * C * sizeof(T)
		double total_bytes = 3.0 * BT * C * sizeof(T);
		double bw = total_bytes / (ms_vec * 1e-3) / 1e9;

		float tol = (sizeof(T) == 2) ? 2e-2f : 1e-3f;
		bool ok_s = matricesEqual(h_scalar, h_ref, BT * C, tol);
		bool ok_v = matricesEqual(h_vec, h_ref, BT * C, tol);
		float speedup = ms_scalar / ms_vec;

		printf("  %-8s %6d %5d  %10.1f %10.1f  %6.2fx %8.1f  %s/%s\n",
				cases[ci].label, BT, C,
				ms_scalar * 1000, ms_vec * 1000,
				speedup, bw,
				ok_s ? "PASS" : "FAIL",
				ok_v ? "PASS" : "FAIL");

		cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
		free(h_a); free(h_b); free(h_ref);
		free(h_scalar); free(h_vec);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
