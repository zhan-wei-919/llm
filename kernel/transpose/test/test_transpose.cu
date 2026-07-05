#include "../Transpose.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>

template<typename T>
__global__ void transpose_naive(T *out, const T *in, int R, int C) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= R * C) return;
	int r = idx / C;
	int c = idx % C;
	out[(long)c * R + r] = in[(long)r * C + c];
}

template<typename T>
void transpose_cpu(float *out, const T *in, int R, int C) {
	for (int r = 0; r < R; ++r)
		for (int c = 0; c < C; ++c)
			out[(long)c * R + r] = static_cast<float>(in[(long)r * C + c]);
}

struct TestCase { const char *label; int R, C; };

static const TestCase cases[] = {
	{ "square",   1024,  1024 },
	{ "tall",     4096,   768 },
	{ "wide",      768,  4096 },
	{ "large",    4096,  4096 },
	{ "big",      8192,  4096 },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== Transpose %s =====\n", type_name);
	printf("  %-8s %6s %5s  %10s %10s  %7s %9s  %s\n",
			"case", "R", "C", "naive(us)", "tiled(us)", "speedup", "BW(GB/s)", "acc");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int R = cases[ci].R, C = cases[ci].C;
		size_t s_data = sizeof(T) * (long)R * C;

		T     *h_in    = (T*)malloc(s_data);
		float *h_ref   = (float*)malloc(sizeof(float) * (long)R * C);
		T     *h_naive = (T*)malloc(s_data);
		T     *h_tiled = (T*)malloc(s_data);

		fill_matrix(h_in, R * C, -1.0f, 1.0f);
		transpose_cpu(h_ref, h_in, R, C);

		T *d_in, *d_out;
		CUDA_CHECK(cudaMalloc(&d_in, s_data));
		CUDA_CHECK(cudaMalloc(&d_out, s_data));
		CUDA_CHECK(cudaMemcpy(d_in, h_in, s_data, cudaMemcpyHostToDevice));

		// naive
		CUDA_CHECK(cudaMemset(d_out, 0, s_data));
		int total = R * C;
		int block_1d = 256;
		int grid_1d = (total + block_1d - 1) / block_1d;
		float ms_naive = 0;
		TIME_MS(ms_naive, 20, 200,
				(transpose_naive<T><<<grid_1d, block_1d>>>(d_out, d_in, R, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_naive, d_out, s_data, cudaMemcpyDeviceToHost));

		// tiled (shared memory)
		CUDA_CHECK(cudaMemset(d_out, 0, s_data));
		float ms_tiled = 0;
		TIME_MS(ms_tiled, 20, 200,
				launch_transpose_forward(d_out, d_in, R, C));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_tiled, d_out, s_data, cudaMemcpyDeviceToHost));

		// 读 + 写 = 2 * R * C * sizeof(T)
		double total_bytes = 2.0 * R * C * sizeof(T);
		double bw = total_bytes / (ms_tiled * 1e-3) / 1e9;

		bool ok_n = matricesEqual(h_naive, h_ref, R * C, 1e-3f);
		bool ok_t = matricesEqual(h_tiled, h_ref, R * C, 1e-3f);
		float speedup = ms_naive / ms_tiled;

		printf("  %-8s %6d %5d  %10.1f %10.1f  %6.2fx %8.1f  %s/%s\n",
				cases[ci].label, R, C,
				ms_naive * 1000, ms_tiled * 1000,
				speedup, bw,
				ok_n ? "PASS" : "FAIL",
				ok_t ? "PASS" : "FAIL");

		cudaFree(d_in); cudaFree(d_out);
		free(h_in); free(h_ref); free(h_naive); free(h_tiled);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
