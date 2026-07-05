#include "../RoPE.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// 标准 RoPE cos/sin 表：cos_t[pos][i] = cos(pos * theta^(-2i/HS))，sin 同理。
// 形状 [max_seq, HS/2]，float。
static void build_rope_tables(
		std::vector<float> &cos_t, std::vector<float> &sin_t,
		int max_seq, int HS, float theta
) {
	int half = HS / 2;
	cos_t.resize((size_t)max_seq * half);
	sin_t.resize((size_t)max_seq * half);
	for (int pos = 0; pos < max_seq; ++pos) {
		for (int i = 0; i < half; ++i) {
			float freq  = 1.0f / powf(theta, (2.0f * i) / HS);
			float angle = pos * freq;
			cos_t[(size_t)pos * half + i] = cosf(angle);
			sin_t[(size_t)pos * half + i] = sinf(angle);
		}
	}
}

// CPU 参照：对 x [seq_len, NH, HS] 逐 head 做旋转，pos0 是起始绝对位置。
template<typename T>
void rope_cpu(
		float *out, const T *x,
		const float *cos_t, const float *sin_t,
		int seq_len, int NH, int HS, int pos0
) {
	int half = HS / 2;
	for (int t = 0; t < seq_len; ++t) {
		for (int h = 0; h < NH; ++h) {
			const T *head = x + ((size_t)t * NH + h) * HS;
			float   *o    = out + ((size_t)t * NH + h) * HS;
			for (int i = 0; i < half; ++i) {
				float c  = cos_t[(size_t)(pos0 + t) * half + i];
				float s  = sin_t[(size_t)(pos0 + t) * half + i];
				float x0 = static_cast<float>(head[i]);
				float x1 = static_cast<float>(head[i + half]);
				o[i]        = x0 * c - x1 * s;
				o[i + half] = x0 * s + x1 * c;
			}
		}
	}
}

struct TestCase { const char *label; int seq_len, NH, HS, pos0; };

static const TestCase cases[] = {
	{ "small",   8,   4,  64,   0 },
	{ "gpt2",  128,  12,  64,   0 },
	{ "offset", 16,   8, 128, 100 },  // 续写：pos0 != 0
	{ "llama", 256,  32, 128,   0 },
	{ "hs256",  64,   8, 256,   0 },
};

static constexpr int MAX_SEQ = 4096;
static constexpr float THETA = 10000.0f;

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== RoPE %s =====\n", type_name);
	printf("  %-8s %6s %4s %4s %5s  %10s %9s  %s\n",
			"case", "T", "NH", "HS", "pos0", "time(us)", "BW(GB/s)", "acc");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int seq_len = cases[ci].seq_len;
		int NH      = cases[ci].NH;
		int HS      = cases[ci].HS;
		int pos0    = cases[ci].pos0;
		int half    = HS / 2;

		int N = seq_len * NH * HS;
		size_t bytes = sizeof(T) * N;

		std::vector<float> cos_t, sin_t;
		build_rope_tables(cos_t, sin_t, MAX_SEQ, HS, THETA);

		T     *h_x   = (T*)malloc(bytes);
		float *h_ref = (float*)malloc(sizeof(float) * N);
		T     *h_out = (T*)malloc(bytes);
		fill_matrix(h_x, N, -1.0f, 1.0f);

		rope_cpu(h_ref, h_x, cos_t.data(), sin_t.data(), seq_len, NH, HS, pos0);

		T     *d_x;
		float *d_cos, *d_sin;
		size_t s_tab = sizeof(float) * (size_t)MAX_SEQ * half;
		CUDA_CHECK(cudaMalloc(&d_x, bytes));
		CUDA_CHECK(cudaMalloc(&d_cos, s_tab));
		CUDA_CHECK(cudaMalloc(&d_sin, s_tab));
		CUDA_CHECK(cudaMemcpy(d_cos, cos_t.data(), s_tab, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_sin, sin_t.data(), s_tab, cudaMemcpyHostToDevice));

		// 正确性：从干净的 x 跑一次（kernel 原地改写）
		CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
		launch_rope(d_x, d_cos, d_sin, seq_len, NH, HS, pos0, MAX_SEQ);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_out, d_x, bytes, cudaMemcpyDeviceToHost));

		float tol = (sizeof(T) == 2) ? 5e-2f : 1e-3f;
		bool ok = matricesEqual(h_out, h_ref, N, tol);

		// 计时：原地算子会反复旋转，结果无意义，只测时间
		CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
		float ms = 0;
		TIME_MS(ms, 10, 100,
				launch_rope(d_x, d_cos, d_sin, seq_len, NH, HS, pos0, MAX_SEQ));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		// 读 x + 写 x = 2N * sizeof(T)（cos/sin 表流量未计）
		double bw = 2.0 * N * sizeof(T) / (ms * 1e-3) / 1e9;

		printf("  %-8s %6d %4d %4d %5d  %10.1f %9.1f  %s\n",
				cases[ci].label, seq_len, NH, HS, pos0,
				ms * 1000, bw,
				ok ? "PASS" : "FAIL");

		if (!ok) {
			float max_err = 0;
			for (int idx = 0; idx < N; ++idx) {
				float diff = fabsf(static_cast<float>(h_out[idx]) - h_ref[idx]);
				if (diff > max_err) max_err = diff;
			}
			printf("           max_err: %.6f\n", max_err);
		}

		cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin);
		free(h_x); free(h_ref); free(h_out);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
