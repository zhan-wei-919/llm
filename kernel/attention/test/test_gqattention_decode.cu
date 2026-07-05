#include "../GQAttention_decode.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// CPU 参考实现：GQA 解码步 attention（单个 query 位置，对整个 kv cache 全看，无 causal）。
// 全程 float 累加作为精度基准。q 头 h 共享 kv 组 g = h / (NH / NKV)。
template<typename T>
void gqa_decode_cpu(
		float *out,           // [NH*HS]
		const T *q,           // [NH*HS]
		const T *k_cache,     // [cur_len, NKV*HS]
		const T *v_cache,     // [cur_len, NKV*HS]
		int cur_len, int NH, int NKV, int HS
) {
	int kv_stride = NKV * HS;
	int GROUP     = NH / NKV;
	float scale   = 1.0f / sqrtf((float)HS);

	for (int h = 0; h < NH; ++h) {
		int g = h / GROUP;
		const T *q_h = q + (size_t)h * HS;

		std::vector<float> scores(cur_len);
		float row_max = -INFINITY;
		for (int j = 0; j < cur_len; ++j) {
			const T *k_j = k_cache + (size_t)j * kv_stride + g * HS;
			float dot = 0.0f;
			for (int d = 0; d < HS; ++d)
				dot += static_cast<float>(q_h[d]) * static_cast<float>(k_j[d]);
			scores[j] = dot * scale;
			if (scores[j] > row_max) row_max = scores[j];
		}

		float sum = 0.0f;
		for (int j = 0; j < cur_len; ++j) {
			scores[j] = expf(scores[j] - row_max);
			sum += scores[j];
		}
		float inv_z = 1.0f / sum;

		float *out_h = out + (size_t)h * HS;
		for (int d = 0; d < HS; ++d) {
			float acc = 0.0f;
			for (int j = 0; j < cur_len; ++j) {
				const T *v_j = v_cache + (size_t)j * kv_stride + g * HS;
				acc += scores[j] * inv_z * static_cast<float>(v_j[d]);
			}
			out_h[d] = acc;
		}
	}
}

struct TestCase { const char *label; int cur_len, NH, NKV, HS; };

static const TestCase cases[] = {
	{ "mha",     64,   4,   4,  32 },  // NKV==NH，退化成普通 MHA
	{ "mqa",     64,   8,   1,  32 },  // NKV==1，退化成 MQA
	{ "gqa",    128,   8,   2,  64 },
	{ "len1",     1,   8,   2,  64 },  // 首步解码：cache 里只有 1 个位置
	{ "long",  2048,   8,   2,  64 },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== GQAttention decode %s =====\n", type_name);
	printf("  %-8s %6s %4s %4s %4s  %10s  %s\n",
			"case", "cur_len", "NH", "NKV", "HS", "time(us)", "accuracy");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int cur_len = cases[ci].cur_len;
		int NH      = cases[ci].NH;
		int NKV     = cases[ci].NKV;
		int HS      = cases[ci].HS;

		size_t s_q     = sizeof(T) * NH  * HS;
		size_t s_cache = sizeof(T) * cur_len * NKV * HS;
		size_t s_out   = s_q;

		T     *h_q       = (T*)malloc(s_q);
		T     *h_k       = (T*)malloc(s_cache);
		T     *h_v       = (T*)malloc(s_cache);
		float *h_ref_out = (float*)malloc(sizeof(float) * NH * HS);
		T     *h_out     = (T*)malloc(s_out);

		fill_matrix(h_q, NH * HS, -0.5f, 0.5f);
		fill_matrix(h_k, cur_len * NKV * HS, -0.5f, 0.5f);
		fill_matrix(h_v, cur_len * NKV * HS, -0.5f, 0.5f);

		gqa_decode_cpu(h_ref_out, h_q, h_k, h_v, cur_len, NH, NKV, HS);

		T *d_q, *d_k, *d_v, *d_out;
		CUDA_CHECK(cudaMalloc(&d_q, s_q));
		CUDA_CHECK(cudaMalloc(&d_k, s_cache));
		CUDA_CHECK(cudaMalloc(&d_v, s_cache));
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMemcpy(d_q, h_q, s_q, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_k, h_k, s_cache, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_v, h_v, s_cache, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms = 0;
		TIME_MS(ms, 5, 50,
				launch_gq_attention_decode(d_out, d_q, d_k, d_v, cur_len, NH, NKV, HS));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_out, d_out, s_out, cudaMemcpyDeviceToHost));

		float tol = (sizeof(T) == 2) ? 5e-2f : 2e-2f;
		int out_size = NH * HS;
		bool ok_out = matricesEqual(h_out, h_ref_out, out_size, tol);

		printf("  %-8s %6d %4d %4d %4d  %10.1f  out=%s\n",
				cases[ci].label, cur_len, NH, NKV, HS,
				ms * 1000,
				ok_out ? "PASS" : "FAIL");

		if (!ok_out) {
			float max_err = 0;
			for (int idx = 0; idx < out_size; ++idx) {
				float diff = fabsf(static_cast<float>(h_out[idx]) - h_ref_out[idx]);
				if (diff > max_err) max_err = diff;
			}
			printf("           max_err: out=%.6f\n", max_err);
		}

		cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
		free(h_q); free(h_k); free(h_v); free(h_ref_out); free(h_out);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
