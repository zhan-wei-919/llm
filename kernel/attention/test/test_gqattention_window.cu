#include "../Attention.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int NH = 32, NKV = 4, HS = 64;
constexpr int QS = NH * HS, KS = NKV * HS;

struct ErrorStats {
	float max_error = 0.0f;
	double error_sum = 0.0;
	int above = 0, count = 0;

	void add(float got, float want) {
		float error = std::fabs(got - want);
		max_error = std::max(max_error, error);
		error_sum += error;
		above += error > 2e-2f;
		count++;
	}
};

static float value_at(int pos, int index) {
	return ((pos * 7 + index * 3) % 31 - 15) / 32.0f;
}

static float batch_value_at(int batch, int pos, int index) {
	return ((batch * 13 + pos * 7 + index * 3) % 61 - 30) / 32.0f;
}

// 用零 Q/K 产生均匀权重，精确检查窗口下界、causal 上界、非零位置和乱序页表。
static void run_case(int total, int start, int window_size, ErrorStats &stats) {
	using T = __nv_bfloat16;
	int seq_len = start + total;
	int blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	std::vector<int> table(blocks);
	for (int i = 0; i < blocks; ++i) table[i] = blocks - i - 1;
	std::vector<T> v((size_t)blocks * KV_BLOCK_SIZE * KS);
	std::vector<float> prefix((size_t)(seq_len + 1) * KS, 0.0f);
	for (int p = 0; p < seq_len; ++p) {
		int dst = (table[p / KV_BLOCK_SIZE] * KV_BLOCK_SIZE + p % KV_BLOCK_SIZE) * KS;
		for (int i = 0; i < KS; ++i) {
			float x = value_at(p, i);
			v[dst + i] = __float2bfloat16_rn(x);
			prefix[(size_t)(p + 1) * KS + i] = prefix[(size_t)p * KS + i] + x;
		}
	}

	T *q, *k_pool, *v_pool, *out;
	int *d_pos, *d_table;
	cudaMalloc(&q, (size_t)total * QS * sizeof(T));
	cudaMalloc(&k_pool, v.size() * sizeof(T));
	cudaMalloc(&v_pool, v.size() * sizeof(T));
	cudaMalloc(&out, (size_t)total * QS * sizeof(T));
	cudaMalloc(&d_pos, sizeof(int));
	cudaMalloc(&d_table, blocks * sizeof(int));
	cudaMemset(q, 0, (size_t)total * QS * sizeof(T));
	cudaMemset(k_pool, 0, v.size() * sizeof(T));
	cudaMemcpy(v_pool, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, &start, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_table, table.data(), blocks * sizeof(int), cudaMemcpyHostToDevice);
	launch_attention(out, q, k_pool, v_pool, nullptr, d_pos, d_table,
		1, NH, NKV, HS, blocks, total, window_size, 0);
	cudaDeviceSynchronize();

	std::vector<T> got((size_t)total * QS);
	cudaMemcpy(got.data(), out, got.size() * sizeof(T), cudaMemcpyDeviceToHost);
	for (int t = 0; t < total; ++t) {
		int pos = start + t;
		int begin = window_size < 0 ? 0 : std::max(0, pos + 1 - window_size);
		float inv = 1.0f / (pos - begin + 1);
		for (int h = 0; h < NH; ++h) {
			int g = h / (NH / NKV);
			for (int d = 0; d < HS; ++d) {
				int i = g * HS + d;
				float want = (prefix[(size_t)(pos + 1) * KS + i] - prefix[(size_t)begin * KS + i]) * inv;
				stats.add(__bfloat162float(got[(size_t)t * QS + h * HS + d]), want);
			}
		}
	}
	cudaFree(q); cudaFree(k_pool); cudaFree(v_pool); cudaFree(out);
	cudaFree(d_pos); cudaFree(d_table);
}

// 用 ragged packed batch 检查 CTA 不跨序列，且每个序列使用自己的位置和页表。
static void run_multibatch_case(const std::vector<int> &lens, const std::vector<int> &starts, int window_size, ErrorStats &stats) {
	using T = __nv_bfloat16;
	int B = (int)lens.size(), total = 0, W = 0;
	std::vector<int> cu(B + 1);
	for (int b = 0; b < B; ++b) {
		total += lens[b];
		cu[b + 1] = total;
		W = std::max(W, (starts[b] + lens[b] + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE);
	}
	int num_blocks = B * W;
	std::vector<int> table((size_t)B * W);
	for (int i = 0; i < num_blocks; ++i) table[i] = num_blocks - i - 1;
	std::vector<T> q((size_t)total * QS, T(0.0f));
	std::vector<T> k_pool((size_t)num_blocks * KV_BLOCK_SIZE * KS, T(0.0f)), v_pool(k_pool.size());
	std::vector<std::vector<float>> prefix(B);
	for (int b = 0; b < B; ++b) {
		int seq_len = starts[b] + lens[b];
		prefix[b].resize((size_t)(seq_len + 1) * KS);
		for (int p = 0; p < seq_len; ++p) {
			int phys = table[(size_t)b * W + p / KV_BLOCK_SIZE];
			int dst = (phys * KV_BLOCK_SIZE + p % KV_BLOCK_SIZE) * KS;
			for (int i = 0; i < KS; ++i) {
				float x = batch_value_at(b, p, i);
				v_pool[dst + i] = T(x);
				prefix[b][(size_t)(p + 1) * KS + i] = prefix[b][(size_t)p * KS + i] + x;
			}
		}
	}

	T *d_q, *d_k, *d_v, *d_out;
	int *d_cu, *d_pos, *d_table;
	cudaMalloc(&d_q, q.size() * sizeof(T)); cudaMalloc(&d_k, k_pool.size() * sizeof(T));
	cudaMalloc(&d_v, v_pool.size() * sizeof(T)); cudaMalloc(&d_out, q.size() * sizeof(T));
	cudaMalloc(&d_cu, cu.size() * sizeof(int));
	cudaMalloc(&d_pos, starts.size() * sizeof(int)); cudaMalloc(&d_table, table.size() * sizeof(int));
	cudaMemcpy(d_q, q.data(), q.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_k, k_pool.data(), k_pool.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v_pool.data(), v_pool.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cu, cu.data(), cu.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, starts.data(), starts.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_table, table.data(), table.size() * sizeof(int), cudaMemcpyHostToDevice);
	launch_attention(d_out, d_q, d_k, d_v, d_cu, d_pos, d_table,
		B, NH, NKV, HS, W, total, window_size, 0);
	cudaDeviceSynchronize();

	std::vector<T> got(q.size());
	cudaMemcpy(got.data(), d_out, got.size() * sizeof(T), cudaMemcpyDeviceToHost);
	for (int b = 0; b < B; ++b) {
		for (int t = 0; t < lens[b]; ++t) {
			int pos = starts[b] + t;
			int begin = window_size < 0 ? 0 : std::max(0, pos + 1 - window_size);
			float inv = 1.0f / (pos - begin + 1);
			int row = cu[b] + t;
			for (int h = 0; h < NH; ++h) {
				int g = h / (NH / NKV);
				for (int d = 0; d < HS; ++d) {
					int i = g * HS + d;
					float want = (prefix[b][(size_t)(pos + 1) * KS + i] - prefix[b][(size_t)begin * KS + i]) * inv;
					stats.add((float)got[(size_t)row * QS + h * HS + d], want);
				}
			}
		}
	}
	cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
	cudaFree(d_cu); cudaFree(d_pos); cudaFree(d_table);
}

// 用非零 Q/K/V 的 CPU softmax 参考结果检查非 tile 对齐窗口的数值。
static void run_random_case(ErrorStats &stats) {
	using T = __nv_bfloat16;
	constexpr int total = 129, start = 511, window_size = 33;
	int seq_len = start + total;
	int blocks = (seq_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	std::vector<int> table(blocks);
	for (int i = 0; i < blocks; ++i) table[i] = blocks - i - 1;
	std::vector<T> q((size_t)total * QS), k((size_t)seq_len * KS), v((size_t)seq_len * KS);
	std::vector<T> k_pool((size_t)blocks * KV_BLOCK_SIZE * KS), v_pool(k_pool.size());
	for (size_t i = 0; i < q.size(); ++i) q[i] = T(((int)(i * 13 % 29) - 14) / 64.0f);
	for (int p = 0; p < seq_len; ++p) {
		int dst = (table[p / KV_BLOCK_SIZE] * KV_BLOCK_SIZE + p % KV_BLOCK_SIZE) * KS;
		for (int i = 0; i < KS; ++i) {
			k[(size_t)p * KS + i] = T(((p * 7 + i * 11) % 31 - 15) / 64.0f);
			v[(size_t)p * KS + i] = T(((p * 17 + i * 5) % 37 - 18) / 64.0f);
			k_pool[dst + i] = k[(size_t)p * KS + i];
			v_pool[dst + i] = v[(size_t)p * KS + i];
		}
	}

	T *d_q, *d_k, *d_v, *d_out;
	int *d_pos, *d_table;
	cudaMalloc(&d_q, q.size() * sizeof(T)); cudaMalloc(&d_k, k_pool.size() * sizeof(T));
	cudaMalloc(&d_v, v_pool.size() * sizeof(T)); cudaMalloc(&d_out, q.size() * sizeof(T));
	cudaMalloc(&d_pos, sizeof(int)); cudaMalloc(&d_table, blocks * sizeof(int));
	cudaMemcpy(d_q, q.data(), q.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_k, k_pool.data(), k_pool.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v_pool.data(), v_pool.size() * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pos, &start, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_table, table.data(), blocks * sizeof(int), cudaMemcpyHostToDevice);
	launch_attention(d_out, d_q, d_k, d_v, nullptr, d_pos, d_table,
		1, NH, NKV, HS, blocks, total, window_size, 0);
	cudaDeviceSynchronize();
	std::vector<T> got(q.size());
	cudaMemcpy(got.data(), d_out, got.size() * sizeof(T), cudaMemcpyDeviceToHost);

	std::vector<float> scores(window_size);
	for (int t = 0; t < total; ++t) {
		int pos = start + t, begin = pos + 1 - window_size;
		for (int h = 0; h < NH; ++h) {
			int g = h / (NH / NKV);
			float row_max = -INFINITY;
			for (int p = begin; p <= pos; ++p) {
				float dot = 0.0f;
				for (int d = 0; d < HS; ++d)
					dot += (float)q[(size_t)t * QS + h * HS + d] * (float)k[(size_t)p * KS + g * HS + d];
					scores[p - begin] = dot / std::sqrt((float)HS);
				row_max = std::max(row_max, scores[p - begin]);
			}
			float sum = 0.0f;
			for (float &score : scores) {score = std::exp(score - row_max); sum += score;}
			for (int d = 0; d < HS; ++d) {
				float want = 0.0f;
				for (int p = begin; p <= pos; ++p)
					want += scores[p - begin] / sum * (float)v[(size_t)p * KS + g * HS + d];
				stats.add((float)got[(size_t)t * QS + h * HS + d], want);
			}
		}
	}
	cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
	cudaFree(d_pos); cudaFree(d_table);
}

int main() {
	ErrorStats stats;
	run_case(7, 31, -1, stats);
	run_case(129, 63, -1, stats);
	run_case(1, 2047, 1024, stats);
	run_case(127, 63, 1, stats);
	run_case(128, 257, 17, stats);
	run_case(129, 511, 32, stats);
	run_case(129, 1200, 1024, stats);
	run_case(255, 37, 33, stats);
	run_case(256, 1023, 64, stats);
	run_case(257, 513, -1, stats);
	run_case(512, 1023, 127, stats);
	run_multibatch_case({1,17,32,33}, {0,15,63,511}, -1, stats);
	run_multibatch_case({80,80}, {0,31}, -1, stats);
	run_multibatch_case({128,129}, {0,31}, -1, stats);
	run_multibatch_case({7,31,32,33,65}, {31,47,127,511,1023}, 33, stats);
	run_multibatch_case({65,127,129}, {7,255,511}, 127, stats);
	run_multibatch_case(std::vector<int>(17, 1), {0,17,33,65,97,129,161,193,225,257,289,321,353,385,417,449,511}, 1024, stats);
	run_random_case(stats);
	assert(stats.above == 0);
	std::printf("test_gqattention_window PASS max=%.7f mean=%.7f count=%d\n",
		stats.max_error, stats.error_sum / stats.count, stats.count);
}
