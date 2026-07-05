#include "../Reduce.cuh"
#include "../../utils.cuh"
#include <cub/cub.cuh>
#include <cstdio>
#include <cfloat>

// ---- 用你的 block_sum/block_max 做 reduce 的 kernel ----

template<typename T, int BLOCK>
__global__ void reduce_sum_ours(float *out, const T *in, int C) {
	int row = blockIdx.x;
	float sum = 0.0f;
	for (int c = threadIdx.x; c < C; c += BLOCK)
		sum += static_cast<float>(in[row * C + c]);
	sum = block_sum(sum);
	if (threadIdx.x == 0) out[row] = sum;
}

template<typename T, int BLOCK>
__global__ void reduce_max_ours(float *out, const T *in, int C) {
	int row = blockIdx.x;
	float mx = (threadIdx.x < C) ? static_cast<float>(in[row * C + threadIdx.x]) : -INFINITY;
	for (int c = threadIdx.x + BLOCK; c < C; c += BLOCK)
		mx = fmaxf(mx, static_cast<float>(in[row * C + c]));
	mx = block_max(mx);
	if (threadIdx.x == 0) out[row] = mx;
}

// ---- CUB 版本 ----

template<typename T, int BLOCK>
__global__ void reduce_sum_cub(float *out, const T *in, int C) {
	using BlockReduce = cub::BlockReduce<float, BLOCK>;
	__shared__ typename BlockReduce::TempStorage temp;

	int row = blockIdx.x;
	float sum = 0.0f;
	for (int c = threadIdx.x; c < C; c += BLOCK)
		sum += static_cast<float>(in[row * C + c]);
	sum = BlockReduce(temp).Sum(sum);
	if (threadIdx.x == 0) out[row] = sum;
}

template<typename T, int BLOCK>
__global__ void reduce_max_cub(float *out, const T *in, int C) {
	using BlockReduce = cub::BlockReduce<float, BLOCK>;
	__shared__ typename BlockReduce::TempStorage temp;

	int row = blockIdx.x;
	float mx = -INFINITY;
	for (int c = threadIdx.x; c < C; c += BLOCK)
		mx = fmaxf(mx, static_cast<float>(in[row * C + c]));
	mx = BlockReduce(temp).Reduce(mx, ::cuda::maximum<float>{});
	if (threadIdx.x == 0) out[row] = mx;
}

// ---- 测试框架 ----

struct TestCase { const char *label; int rows, C; };

static const TestCase cases[] = {
	{ "small",    1024,   768  },
	{ "gpt2",    4096,   768  },
	{ "llama",   4096,   4096 },
	{ "big",     8192,   4096 },
	{ "wide",    1024,  16384 },
};

static constexpr int BLOCK = 256;

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== Reduce %s =====\n", type_name);

	// --- SUM ---
	printf("\n  [SUM]\n");
	printf("  %-8s %6s %6s  %10s %10s  %7s  %s\n",
			"case", "rows", "C", "ours(us)", "CUB(us)", "ratio", "acc");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int rows = cases[ci].rows, C = cases[ci].C;
		size_t s_in  = sizeof(T) * rows * C;
		size_t s_out = sizeof(float) * rows;

		T     *h_in  = (T*)malloc(s_in);
		float *h_ours = (float*)malloc(s_out);
		float *h_cub  = (float*)malloc(s_out);
		fill_matrix(h_in, rows * C, -1.0f, 1.0f);

		T *d_in; float *d_out;
		CUDA_CHECK(cudaMalloc(&d_in, s_in));
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMemcpy(d_in, h_in, s_in, cudaMemcpyHostToDevice));

		// ours
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_ours = 0;
		TIME_MS(ms_ours, 20, 200,
				(reduce_sum_ours<T, BLOCK><<<rows, BLOCK>>>(d_out, d_in, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_ours, d_out, s_out, cudaMemcpyDeviceToHost));

		// CUB
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_cub = 0;
		TIME_MS(ms_cub, 20, 200,
				(reduce_sum_cub<T, BLOCK><<<rows, BLOCK>>>(d_out, d_in, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_cub, d_out, s_out, cudaMemcpyDeviceToHost));

		bool ok = matricesEqual(h_ours, h_cub, rows, 1e-3f);
		float ratio = ms_cub / ms_ours;

		printf("  %-8s %6d %6d  %10.1f %10.1f  %6.2fx  %s\n",
				cases[ci].label, rows, C,
				ms_ours * 1000, ms_cub * 1000, ratio,
				ok ? "PASS" : "FAIL");

		cudaFree(d_in); cudaFree(d_out);
		free(h_in); free(h_ours); free(h_cub);
	}

	// --- MAX ---
	printf("\n  [MAX]\n");
	printf("  %-8s %6s %6s  %10s %10s  %7s  %s\n",
			"case", "rows", "C", "ours(us)", "CUB(us)", "ratio", "acc");

	for (int ci = 0; ci < num; ++ci) {
		int rows = cases[ci].rows, C = cases[ci].C;
		size_t s_in  = sizeof(T) * rows * C;
		size_t s_out = sizeof(float) * rows;

		T     *h_in   = (T*)malloc(s_in);
		float *h_ours = (float*)malloc(s_out);
		float *h_cub  = (float*)malloc(s_out);
		fill_matrix(h_in, rows * C, -1.0f, 1.0f);

		T *d_in; float *d_out;
		CUDA_CHECK(cudaMalloc(&d_in, s_in));
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMemcpy(d_in, h_in, s_in, cudaMemcpyHostToDevice));

		// ours
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_ours = 0;
		TIME_MS(ms_ours, 20, 200,
				(reduce_max_ours<T, BLOCK><<<rows, BLOCK>>>(d_out, d_in, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_ours, d_out, s_out, cudaMemcpyDeviceToHost));

		// CUB
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_cub = 0;
		TIME_MS(ms_cub, 20, 200,
				(reduce_max_cub<T, BLOCK><<<rows, BLOCK>>>(d_out, d_in, C)));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_cub, d_out, s_out, cudaMemcpyDeviceToHost));

		bool ok = matricesEqual(h_ours, h_cub, rows, 1e-3f);
		float ratio = ms_cub / ms_ours;

		printf("  %-8s %6d %6d  %10.1f %10.1f  %6.2fx  %s\n",
				cases[ci].label, rows, C,
				ms_ours * 1000, ms_cub * 1000, ratio,
				ok ? "PASS" : "FAIL");

		cudaFree(d_in); cudaFree(d_out);
		free(h_in); free(h_ours); free(h_cub);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
