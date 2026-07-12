// 多层 KV pool 隔离测试:
// 同一次 prepare 共享页表，两层分别 scatter 不同 KV，验证数据和 attention 输出互不污染。
//
// 编译: nvcc -O2 -arch=sm_120 test_kv_pool_layers.cu -o test_kv_pool_layers
#include "../Engine.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int NUM_LAYERS = 2;
constexpr int NH = 1, NKV = 1, HS = 4;
constexpr int QS = NH * HS, KS = NKV * HS;
constexpr int TOKENS = 3;
constexpr int MAX_SEQS = 1, MAX_SEQ_LEN = KV_BLOCK_SIZE;

static void check_close(float got, float want) {
	if (std::fabs(got - want) >= 1e-6f)
		std::fprintf(stderr, "mismatch: got=%f want=%f\n", got, want);
	assert(std::fabs(got - want) < 1e-6f);
}

int main() {
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	const size_t layer_bytes = (size_t)KV_BLOCK_SIZE * KS * sizeof(float);
	const size_t capacity = NUM_LAYERS * layer_bytes;
	void *k_base, *v_base;
	cudaMalloc(&k_base, capacity);
	cudaMalloc(&v_base, capacity);

	Arena arena(/*max_tensors=*/3); // Engine 在 finalize 前登记 meta/cos/sin
	Engine engine(arena, NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN);
	arena.finalize();
	KV_Pool pool(k_base, v_base, Dtype::F32, KS, capacity,
	             NUM_LAYERS, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);

	std::vector<float> h_q(TOKENS * QS, 0.0f);
	std::vector<float> h_k0(TOKENS * KS, 1.0f);
	std::vector<float> h_v0(TOKENS * KS, 2.0f);
	std::vector<float> h_k1(TOKENS * KS, 5.0f);
	std::vector<float> h_v1(TOKENS * KS, 7.0f);
	float *d_q, *d_k0, *d_v0, *d_k1, *d_v1, *d_out0, *d_out1;
	cudaMalloc(&d_q, TOKENS * QS * sizeof(float));
	cudaMalloc(&d_k0, TOKENS * KS * sizeof(float));
	cudaMalloc(&d_v0, TOKENS * KS * sizeof(float));
	cudaMalloc(&d_k1, TOKENS * KS * sizeof(float));
	cudaMalloc(&d_v1, TOKENS * KS * sizeof(float));
	cudaMalloc(&d_out0, TOKENS * QS * sizeof(float));
	cudaMalloc(&d_out1, TOKENS * QS * sizeof(float));
	cudaMemcpyAsync(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_k0, h_k0.data(), h_k0.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_v0, h_v0.data(), h_v0.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_k1, h_k1.data(), h_k1.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_v1, h_v1.data(), h_v1.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

	int slot = engine.alloc_seq();
	int slots[] = {slot}, lens[] = {TOKENS};
	engine.prepare(slots, 1, lens, stream);
	engine.forward_layer(0, d_q, d_k0, d_v0, d_out0, stream);
	engine.forward_layer(1, d_q, d_k1, d_v1, d_out1, stream);

	std::vector<float> cached_k0(TOKENS * KS), cached_v0(TOKENS * KS);
	std::vector<float> cached_k1(TOKENS * KS), cached_v1(TOKENS * KS);
	std::vector<float> out0(TOKENS * QS), out1(TOKENS * QS);
	cudaMemcpyAsync(cached_k0.data(), pool.k_base(0), cached_k0.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(cached_v0.data(), pool.v_base(0), cached_v0.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(cached_k1.data(), pool.k_base(1), cached_k1.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(cached_v1.data(), pool.v_base(1), cached_v1.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(out0.data(), d_out0, out0.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(out1.data(), d_out1, out1.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	for (size_t i = 0; i < cached_k0.size(); ++i) {
		check_close(cached_k0[i], 1.0f);
		check_close(cached_v0[i], 2.0f);
		check_close(cached_k1[i], 5.0f);
		check_close(cached_v1[i], 7.0f);
	}
	for (size_t i = 0; i < out0.size(); ++i) {
		check_close(out0[i], 2.0f);
		check_close(out1[i], 7.0f);
	}
	assert(pool.seq_len(slot) == TOKENS); // prepare 只记账一次

	cudaFree(d_q); cudaFree(d_k0); cudaFree(d_v0); cudaFree(d_k1); cudaFree(d_v1);
	cudaFree(d_out0); cudaFree(d_out1);
	cudaFree(k_base); cudaFree(v_base); cudaStreamDestroy(stream);
	printf("test_kv_pool_layers PASS\n");
	return 0;
}
