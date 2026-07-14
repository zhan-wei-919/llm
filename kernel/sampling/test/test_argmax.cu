#include "../Argmax.h"
#include <cassert>
#include <cstdio>
#include <vector>

template<typename T>
T make_value(float value);

template<>
float make_value<float>(float value) { return value; }

template<>
half make_value<half>(float value) { return __float2half(value); }

template<>
__nv_bfloat16 make_value<__nv_bfloat16>(float value) {
	return __float2bfloat16(value);
}

template<typename T>
void test_type() {
	constexpr int B = 3;
	constexpr int TOTAL = 6;
	constexpr int VOCAB = 513;
	const int cu_seqlens[B + 1] = {0, 2, 5, 6};
	std::vector<T> logits((size_t)TOTAL * VOCAB, make_value<T>(-1000.0f));
	logits[(size_t)1 * VOCAB + 7] = make_value<T>(50.0f);
	logits[(size_t)4 * VOCAB + 3] = make_value<T>(99.0f);
	logits[(size_t)4 * VOCAB + 9] = make_value<T>(99.0f);
	logits[(size_t)5 * VOCAB + 512] = make_value<T>(25.0f);

	T *d_logits;
	int *d_cu_seqlens, *d_tokens;
	cudaMalloc(&d_logits, logits.size() * sizeof(T));
	cudaMalloc(&d_cu_seqlens, sizeof(cu_seqlens));
	cudaMalloc(&d_tokens, B * sizeof(int));
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	cudaMemcpyAsync(
		d_logits, logits.data(), logits.size() * sizeof(T),
		cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(
		d_cu_seqlens, cu_seqlens, sizeof(cu_seqlens),
		cudaMemcpyHostToDevice, stream);
	launch_argmax_last_token(
		d_tokens, d_logits, dtype_of<T>::value, d_cu_seqlens, B, VOCAB, stream);
	int tokens[B];
	cudaMemcpyAsync(tokens, d_tokens, sizeof(tokens), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	assert(tokens[0] == 7);
	assert(tokens[1] == 3);
	assert(tokens[2] == 512);
	cudaStreamDestroy(stream);
	cudaFree(d_tokens);
	cudaFree(d_cu_seqlens);
	cudaFree(d_logits);
}

int main() {
	test_type<float>();
	test_type<half>();
	test_type<__nv_bfloat16>();
	std::printf("test_argmax PASS: f32 + f16 + bf16\n");
	return 0;
}
