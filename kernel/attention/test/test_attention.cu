#include "../Attention.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

template<typename T>
void attention_cpu(
		float *out,       // [B, seq_len, C]
		float *att,       // [B, NH, seq_len, seq_len]
		const T *qkv,    // [B, seq_len, 3C]
		int B, int seq_len, int C, int NH
) {
	int HS = C / NH;
	float scale = 1.0f / sqrtf((float)HS);

	for (int b = 0; b < B; ++b) {
		for (int h = 0; h < NH; ++h) {
			for (int i = 0; i < seq_len; ++i) {
				const T *q_i = qkv + (size_t)(b * seq_len + i) * 3 * C + 0 * C + h * HS;

				float row_max = -INFINITY;
				float scores[MAX_SEQ_LEN];
				for (int j = 0; j <= i; ++j) {
					const T *k_j = qkv + (size_t)(b * seq_len + j) * 3 * C + 1 * C + h * HS;
					float dot = 0.0f;
					for (int d = 0; d < HS; ++d)
						dot += static_cast<float>(q_i[d]) * static_cast<float>(k_j[d]);
					scores[j] = dot * scale;
					if (scores[j] > row_max) row_max = scores[j];
				}

				float sum = 0.0f;
				for (int j = 0; j <= i; ++j) {
					scores[j] = expf(scores[j] - row_max);
					sum += scores[j];
				}
				float inv_z = 1.0f / sum;

				for (int j = 0; j <= i; ++j)
					scores[j] *= inv_z;

				float *out_i = out + (size_t)(b * seq_len + i) * C + h * HS;
				for (int d = 0; d < HS; ++d) {
					float acc = 0.0f;
					for (int j = 0; j <= i; ++j) {
						const T *v_j = qkv + (size_t)(b * seq_len + j) * 3 * C + 2 * C + h * HS;
						acc += scores[j] * static_cast<float>(v_j[d]);
					}
					out_i[d] = acc;
				}

				float *att_i = att + ((size_t)(b * NH + h) * seq_len + i) * seq_len;
				for (int j = 0; j < seq_len; ++j)
					att_i[j] = (j <= i) ? scores[j] : 0.0f;
			}
		}
	}
}

struct TestCase { const char *label; int B, seq_len, C, NH; };

static const TestCase cases[] = {
	{ "tiny",    1,   32,   64,   2 },
	{ "small",   1,   64,  128,   4 },
	{ "gpt2",    1,  128,  768,  12 },
	{ "multi-b", 2,   64,  256,   4 },
	{ "long",    1,  512,  256,   4 },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== Attention %s =====\n", type_name);
	printf("  %-8s %3s %5s %5s %3s  %10s  %s\n",
			"case", "B", "T", "C", "NH", "time(us)", "accuracy");

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int B       = cases[ci].B;
		int seq_len = cases[ci].seq_len;
		int C       = cases[ci].C;
		int NH      = cases[ci].NH;

		size_t s_qkv = sizeof(T) * B * seq_len * 3 * C;
		size_t s_out = sizeof(T) * B * seq_len * C;
		size_t s_att = sizeof(T) * B * NH * seq_len * seq_len;

		T     *h_qkv    = (T*)malloc(s_qkv);
		float *h_ref_out = (float*)malloc(sizeof(float) * B * seq_len * C);
		float *h_ref_att = (float*)malloc(sizeof(float) * B * NH * seq_len * seq_len);
		T     *h_out     = (T*)malloc(s_out);
		T     *h_att     = (T*)malloc(s_att);

		fill_matrix(h_qkv, B * seq_len * 3 * C, -0.5f, 0.5f);

		attention_cpu(h_ref_out, h_ref_att, h_qkv, B, seq_len, C, NH);

		T *d_qkv, *d_out, *d_att;
		CUDA_CHECK(cudaMalloc(&d_qkv, s_qkv));
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMalloc(&d_att, s_att));
		CUDA_CHECK(cudaMemcpy(d_qkv, h_qkv, s_qkv, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		CUDA_CHECK(cudaMemset(d_att, 0, s_att));
		float ms = 0;
		TIME_MS(ms, 5, 50,
				launch_attention_forward(d_out, d_att, d_qkv, B, seq_len, C, NH));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_out, d_out, s_out, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_att, d_att, s_att, cudaMemcpyDeviceToHost));

		float tol = (sizeof(T) == 2) ? 5e-2f : 2e-2f;
		bool ok_out = matricesEqual(h_out, h_ref_out, B * seq_len * C, tol);
		bool ok_att = matricesEqual(h_att, h_ref_att, B * NH * seq_len * seq_len, tol);

		printf("  %-8s %3d %5d %5d %3d  %10.1f  out=%s att=%s\n",
				cases[ci].label, B, seq_len, C, NH,
				ms * 1000,
				ok_out ? "PASS" : "FAIL",
				ok_att ? "PASS" : "FAIL");

		if (!ok_out || !ok_att) {
			int out_size = B * seq_len * C;
			int att_size = B * NH * seq_len * seq_len;
			float max_out_err = 0, max_att_err = 0;
			for (int k = 0; k < out_size; ++k) {
				float diff = fabsf(static_cast<float>(h_out[k]) - h_ref_out[k]);
				if (diff > max_out_err) max_out_err = diff;
			}
			for (int k = 0; k < att_size; ++k) {
				float diff = fabsf(static_cast<float>(h_att[k]) - h_ref_att[k]);
				if (diff > max_att_err) max_att_err = diff;
			}
			printf("           max_err: out=%.6f att=%.6f\n", max_out_err, max_att_err);
		}

		cudaFree(d_qkv); cudaFree(d_out); cudaFree(d_att);
		free(h_qkv); free(h_ref_out); free(h_ref_att);
		free(h_out); free(h_att);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
