#include "../TokenEmbedding.h"
#include "../../utils.h"
#include <cassert>
#include <cstdio>
#include <vector>

template<typename T>
static void test_type(const char *name) {
	constexpr int V = 7, C = 13, TOTAL = 5;
	const int ids[TOTAL] = {6, 0, 3, 3, 1};
	std::vector<T> weight(V * C), out(TOTAL * C);
	for (int v = 0; v < V; ++v)
		for (int d = 0; d < C; ++d)
			weight[v * C + d] = static_cast<T>(v * 100 + d);

	T *d_weight, *d_out;
	int *d_ids;
	CUDA_CHECK(cudaMalloc(&d_weight, weight.size() * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&d_out, out.size() * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&d_ids, sizeof(ids)));
	CUDA_CHECK(cudaMemcpy(d_weight, weight.data(), weight.size() * sizeof(T), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_ids, ids, sizeof(ids), cudaMemcpyHostToDevice));
	launch_token_embedding(d_out, d_ids, d_weight, TOTAL, C, cudaStreamPerThread);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(out.data(), d_out, out.size() * sizeof(T), cudaMemcpyDeviceToHost));

	for (int t = 0; t < TOTAL; ++t)
		for (int d = 0; d < C; ++d)
			assert(static_cast<float>(out[t * C + d]) ==
			       static_cast<float>(weight[ids[t] * C + d]));

	cudaFree(d_weight); cudaFree(d_out); cudaFree(d_ids);
	std::printf("test_token_embedding %s PASS\n", name);
}

int main() {
	test_type<float>("f32");
	test_type<half>("f16");
	test_type<__nv_bfloat16>("bf16");
	return 0;
}
