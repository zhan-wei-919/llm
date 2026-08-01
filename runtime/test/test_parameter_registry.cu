#include "../../model/modules/Embedding.h"
#include "../../model/modules/Linear.h"
#include "../../model/modules/RMSNorm.h"
#include "../../model/modules/Residual.h"
#include <cassert>
#include <cstdio>

int main() {
	LLM llm(/*max_tensors=*/16);
	Tensor *token_ids = llm.arena().alloc({8}, Dtype::I32);
	Embedding<float> embedding(llm, token_ids, 8, 32, 16, "model.embed_tokens");
	RMSNorm<float> norm(llm, embedding.out(), 8, 16, 1e-5f, "model.norm");
	Linear<float> with_bias(llm, norm.out(), 8, 16, 12, true, "model.proj");
	Linear<float> no_bias(llm, with_bias.out(), 8, 12, 16, false, "model.out");
	Residual<float> residual(llm, embedding.out(), no_bias.out(), 8, 16);

	const auto &parameters = llm.parameters();
	assert(parameters.size() == 5);
	assert(parameters.at("model.embed_tokens.weight").tensor == embedding.weight());
	assert(parameters.at("model.norm.weight").tensor->shape[0] == 16);
	assert(parameters.at("model.proj.weight").tensor->shape[0] == 16);
	assert(parameters.at("model.proj.weight").tensor->shape[1] == 12);
	assert(parameters.at("model.proj.bias").tensor->shape[0] == 12);
	assert(parameters.at("model.out.weight").tensor->shape[0] == 12);
	assert(parameters.at("model.out.weight").tensor->shape[1] == 16);
	assert(parameters.find("model.out.bias") == parameters.end());
	assert(parameters.find("model.proj") == parameters.end());
	llm.finalize();
	std::printf("test_parameter_registry PASS\n");
	return 0;
}
