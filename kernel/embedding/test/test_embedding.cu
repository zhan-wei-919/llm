#include "../add_pos_embedding.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>

// 标量版 baseline
template<typename T>
__global__ void embedding_scalar(
		T *out, const int *token_ids, const T *token_table, const T *pos_table,
		int B, int seq_len, int C
) {
	int bt = blockIdx.x;
	int seq_pos = bt % seq_len;
	int token_id = token_ids[bt];
	for (int c = threadIdx.x; c < C; c += blockDim.x) {
		float tok = static_cast<float>(token_table[token_id * C + c]);
		float pos = static_cast<float>(pos_table[seq_pos * C + c]);
		out[bt * C + c] = static_cast<T>(tok + pos);
	}
}

// CPU 参照
template<typename T>
void embedding_cpu(
		T *out, const int *token_ids, const T *token_table, const T *pos_table,
		int B, int seq_len, int C
) {
	for (int bt = 0; bt < B * seq_len; ++bt) {
		int seq_pos = bt % seq_len;
		int token_id = token_ids[bt];
		for (int c = 0; c < C; ++c) {
			float tok = static_cast<float>(token_table[token_id * C + c]);
			float pos = static_cast<float>(pos_table[seq_pos * C + c]);
			out[bt * C + c] = static_cast<T>(tok + pos);
		}
	}
}

struct TestCase { const char *label; int B, T, C, V; };

static const TestCase cases[] = {
	{ "small",    1,   128,  768,   50257 },
	{ "gpt2",    4,   1024, 768,   50257 },
	{ "gpt2-lg", 4,   1024, 1280,  50257 },
	{ "llama",   4,   2048, 4096,  32000 },
	{ "big",     8,   2048, 4096,  32000 },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== Embedding %s =====\n", type_name);
	printf("%-10s %4s %5s %5s %6s  %10s %10s  %7s %9s  %s\n",
			"case", "B", "T", "C", "V", "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "acc");

	int num_cases = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num_cases; ++ci) {
		int B = cases[ci].B, Tlen = cases[ci].T, C = cases[ci].C, V = cases[ci].V;
		int BT = B * Tlen;

		size_t s_out   = sizeof(T) * BT * C;
		size_t s_tok   = sizeof(T) * V * C;
		size_t s_pos   = sizeof(T) * Tlen * C;
		size_t s_ids   = sizeof(int) * BT;

		T   *h_tok_table = (T*)malloc(s_tok);
		T   *h_pos_table = (T*)malloc(s_pos);
		int *h_ids       = (int*)malloc(s_ids);
		T   *h_ref       = (T*)malloc(s_out);
		T   *h_scalar    = (T*)malloc(s_out);
		T   *h_vec       = (T*)malloc(s_out);

		fill_matrix(h_tok_table, V * C, -1.0f, 1.0f);
		fill_matrix(h_pos_table, Tlen * C, -0.5f, 0.5f);
		srand(42);
		for (int i = 0; i < BT; ++i) h_ids[i] = rand() % V;

		embedding_cpu(h_ref, h_ids, h_tok_table, h_pos_table, B, Tlen, C);

		T *d_out, *d_tok, *d_pos; int *d_ids;
		CUDA_CHECK(cudaMalloc(&d_out, s_out));
		CUDA_CHECK(cudaMalloc(&d_tok, s_tok));
		CUDA_CHECK(cudaMalloc(&d_pos, s_pos));
		CUDA_CHECK(cudaMalloc(&d_ids, s_ids));
		CUDA_CHECK(cudaMemcpy(d_tok, h_tok_table, s_tok, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_pos, h_pos_table, s_pos, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_ids, h_ids, s_ids, cudaMemcpyHostToDevice));

		// 标量版
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_scalar = 0;
		TIME_MS(ms_scalar, 20, 100,
				embedding_scalar<<<BT, 256>>>(d_out, d_ids, d_tok, d_pos, B, Tlen, C));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_scalar, d_out, s_out, cudaMemcpyDeviceToHost));

		// 向量化版
		CUDA_CHECK(cudaMemset(d_out, 0, s_out));
		float ms_vec = 0;
		TIME_MS(ms_vec, 20, 100,
				launch_embedding_forward(d_out, d_ids, d_tok, d_pos, B, Tlen, C));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_vec, d_out, s_out, cudaMemcpyDeviceToHost));

		// 带宽：读 token_table BT*C + 读 pos_table BT*C + 写 out BT*C = 3*BT*C*sizeof(T)
		// （token_ids 很小忽略不计）
		double total_bytes = 3.0 * BT * C * sizeof(T);
		double bw = total_bytes / (ms_vec * 1e-3) / 1e9;

		bool ok_s = matricesEqual(h_scalar, h_ref, BT * C, 1e-3f);
		bool ok_v = matricesEqual(h_vec, h_ref, BT * C, 1e-3f);
		float speedup = ms_scalar / ms_vec;

		printf("%-10s %4d %5d %5d %6d  %10.1f %10.1f  %6.2fx %8.1f  %s/%s\n",
				cases[ci].label, B, Tlen, C, V,
				ms_scalar * 1000, ms_vec * 1000,
				speedup, bw,
				ok_s ? "PASS" : "FAIL",
				ok_v ? "PASS" : "FAIL");

		cudaFree(d_out); cudaFree(d_tok); cudaFree(d_pos); cudaFree(d_ids);
		free(h_tok_table); free(h_pos_table); free(h_ids);
		free(h_ref); free(h_scalar); free(h_vec);
	}
}

int main() {
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
	test_type<float>("f32");
}
