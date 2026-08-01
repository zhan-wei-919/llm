#include "../../model/modules/Embedding.h"
#include "../../model/modules/LMHead.h"
#include "../Engine.h"
#include <cassert>
#include <cstdio>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int NH = 1, NKV = 1, HS = 16;
constexpr int HIDDEN = NH * HS;
constexpr int VOCAB = 32000;
constexpr int MAX_TOKENS = 8;
constexpr int MAX_SEQS = 4;
constexpr int MAX_SEQ_LEN = 16;
constexpr int BLOCKS = 8;
constexpr int MAX_NEW_TOKENS = 3;

int main() {
	const std::string tokenizer_path = "weight/config/tokenizer.json";
	const int generated_token = tokenize("Hello", tokenizer_path).back();
	const size_t kv_bytes =
		(size_t)BLOCKS * KV_BLOCK_SIZE * NKV * HS * sizeof(float);
	void *k_base, *v_base;
	cudaMalloc(&k_base, kv_bytes);
	cudaMalloc(&v_base, kv_bytes);

	{
		LLM llm(/*max_tensors=*/8);
		Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN, MAX_TOKENS);
		Tensor *token_ids = llm.arena().alloc({MAX_TOKENS}, Dtype::I32);
		Embedding<float> embedding(
			llm, token_ids, MAX_TOKENS, VOCAB, HIDDEN, "model.embed_tokens");
		LMHead<float> lm_head(
			llm, embedding.out(), MAX_TOKENS, HIDDEN, VOCAB, "lm_head");
		llm.finalize();

		std::vector<float> embedding_weight((size_t)VOCAB * HIDDEN, 1.0f);
		std::vector<float> lm_head_weight((size_t)HIDDEN * VOCAB, 0.0f);
		for (int h = 0; h < HIDDEN; ++h)
			lm_head_weight[(size_t)h * VOCAB + generated_token] = 1.0f;
		cudaMemcpy(
			embedding.weight()->ptr,
			embedding_weight.data(),
			embedding.weight()->bytes(),
			cudaMemcpyHostToDevice);
		cudaMemcpy(
			lm_head.weight()->ptr,
			lm_head_weight.data(),
			lm_head.weight()->bytes(),
			cudaMemcpyHostToDevice);

		KV_Pool pool(
			k_base, v_base, Dtype::F32, NKV * HS, kv_bytes,
			/*num_layers=*/1, MAX_SEQS, MAX_SEQ_LEN);
		engine.bind_pool(&pool);

		const std::vector<std::string> prompts = {
			"Hello", "world", "TinyLlama"
		};
		size_t next_input = 0;
		int nonblocking_polls = 0;
		auto receive = [&](bool wait) -> std::optional<std::string> {
			if (wait) {
				if (next_input == 0) return prompts[next_input++];
				return std::nullopt;
			}
			++nonblocking_polls;
			if ((nonblocking_polls == 2 || nonblocking_polls == 4)
			 && next_input < prompts.size())
				return prompts[next_input++];
			return std::nullopt;
		};

		std::vector<int> completion_order;
		std::unordered_map<int, std::string> outputs;
		auto emit = [&](int request_id, std::string generated_text) {
			completion_order.push_back(request_id);
			outputs.emplace(request_id, std::move(generated_text));
		};

		llm.server(
			receive,
			emit,
			MAX_NEW_TOKENS,
			engine,
			pool,
			{MAX_SEQS, MAX_TOKENS, /*eos=*/-1},
			tokenizer_path);

		const std::string expected = detokenize(
			std::vector<int>(MAX_NEW_TOKENS, generated_token),
			tokenizer_path);
		assert(next_input == prompts.size());
		assert(completion_order.size() == prompts.size());
		for (int id = 0; id < (int)prompts.size(); ++id)
			assert(outputs.at(id) == expected);
		assert((int)pool.num_free_blocks() == BLOCKS);
		assert((int)pool.num_free_slots() == MAX_SEQS);
	}

	cudaFree(k_base);
	cudaFree(v_base);
	std::printf("test_llm_server PASS: online inputs + async inference + GPU argmax\n");
	return 0;
}
