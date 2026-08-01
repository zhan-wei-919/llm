#include "../modules/Embedding.h"
#include "../modules/LMHead.h"
#include "../modules/MLP.h"
#include "../modules/RMSNorm.h"
#include "../modules/Residual.h"
#include "../modules/Attention.h"
#include "../../runtime/Engine.h"
#include <cuda_bf16.h>
#include <iostream>
#include <memory>
#include <optional>
#include <poll.h>
#include <string>
#include <unistd.h>
#include <vector>

constexpr int NUM_LAYERS = 22, NH = 32, NKV = 4, HS = 64;
constexpr int HIDDEN = 2048, INTERMEDIATE = 5632, VOCAB = 32000;
constexpr int MAX_SEQS = 8, MAX_BATCHED_TOKENS = 512, MAX_SEQ_LEN = 2048;
constexpr int MAX_NEW_TOKENS = 256, EOS_TOKEN_ID = 2;
constexpr float RMS_EPS = 1e-5f, ROPE_THETA = 10000.0f;
const std::string WEIGHT_PATH = "weight/weight/model.safetensors";
const std::string TOKENIZER_PATH = "weight/config/tokenizer.json";

template<typename T>
class TinyLlamaBlock {
public:
	TinyLlamaBlock(LLM &llm, Engine &engine, Tensor *input, int layer)
	: input_norm_(llm, input, MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, prefix(layer) + ".input_layernorm")
	, self_attn_(llm, engine, input_norm_.out(), MAX_BATCHED_TOKENS, HIDDEN, layer, false, false, -1, prefix(layer) + ".self_attn")
	, attention_residual_(llm, input, self_attn_.out(), MAX_BATCHED_TOKENS, HIDDEN)
	, post_attention_norm_(llm, attention_residual_.out(), MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, prefix(layer) + ".post_attention_layernorm")
	, mlp_(llm, post_attention_norm_.out(), MAX_BATCHED_TOKENS, HIDDEN, INTERMEDIATE, false, prefix(layer) + ".mlp")
	, mlp_residual_(llm, attention_residual_.out(), mlp_.out(), MAX_BATCHED_TOKENS, HIDDEN) {}

	Tensor *out() {return mlp_residual_.out();}

private:
	static std::string prefix(int layer) {return "model.layers." + std::to_string(layer);}
	RMSNorm<T> input_norm_;
	Attention<T> self_attn_;
	Residual<T> attention_residual_;
	RMSNorm<T> post_attention_norm_;
	MLP<T> mlp_;
	Residual<T> mlp_residual_;
};

int main() {
	using T = __nv_bfloat16;
	LLM llm(512);
	Engine engine(llm.arena(), NH, NKV, HS, MAX_SEQS, MAX_SEQ_LEN, MAX_BATCHED_TOKENS);
	Tensor *token_ids = llm.arena().alloc({MAX_BATCHED_TOKENS}, Dtype::I32);
	Embedding<T> embedding(llm, token_ids, MAX_BATCHED_TOKENS, VOCAB, HIDDEN, "model.embed_tokens");
	std::vector<std::unique_ptr<TinyLlamaBlock<T>>> layers;
	layers.reserve(NUM_LAYERS);
	Tensor *hidden = embedding.out();
	for (int layer = 0; layer < NUM_LAYERS; ++layer) {
		layers.emplace_back(std::make_unique<TinyLlamaBlock<T>>(llm, engine, hidden, layer));
		hidden = layers.back()->out();
	}
	RMSNorm<T> final_norm(llm, hidden, MAX_BATCHED_TOKENS, HIDDEN, RMS_EPS, "model.norm");
	LMHead<T> lm_head(llm, final_norm.out(), MAX_BATCHED_TOKENS, HIDDEN, VOCAB, "lm_head");

	llm.finalize();
	KVAlloc kv = llm.arena().alloc_kv_pool();
	KV_Pool pool(kv.k_base, kv.v_base, Dtype::BF16, NKV * HS, kv.bytes_each, NUM_LAYERS, MAX_SEQS, MAX_SEQ_LEN);
	engine.bind_pool(&pool);
	engine.init_rope(llm.stream(), ROPE_THETA);
	llm.load_safetensor(WEIGHT_PATH, [](const std::string &name) {
		return name == "lm_head.weight" || name.find("_proj.weight") != std::string::npos;
	});

	auto receive = [](bool wait) -> std::optional<std::string> {
		pollfd fd{STDIN_FILENO, POLLIN, 0};
		if (poll(&fd, 1, wait ? -1 : 0) == 0) return std::nullopt;
		std::string input;
		if (!std::getline(std::cin, input)) return std::nullopt;
		return "<|user|>\n" + input + "</s>\n<|assistant|>";
	};
	auto emit = [](int request_id, std::string output) {
		std::cout << request_id << ": " << output << '\n' << std::flush;
	};
	llm.server(receive, emit, MAX_NEW_TOKENS, engine, pool, {MAX_SEQS, MAX_BATCHED_TOKENS, EOS_TOKEN_ID}, TOKENIZER_PATH);
	return 0;
}
