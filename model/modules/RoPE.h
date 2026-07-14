#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"
#include "../../kernel/embedding/RoPE.h"

template<typename T>
class RoPE : public Module {
public:
	RoPE(LLM &llm, Engine &engine, Tensor *q, Tensor *k)
	: engine_(engine), q_(q), k_(k){attach(llm, *this, {q_, k_}, {q_, k_});}

	void forward(const GraphShape &, cudaStream_t stream) {
		const float *cos_table = static_cast<const float*>(engine_.cos_table()->ptr);
		const float *sin_table = static_cast<const float*>(engine_.sin_table()->ptr);
		engine_.apple_rope(q_->ptr, k_->ptr, cos_table, sin_table, stream);
	}

	// rope原地操作没有out()
private:
	Engine &engine_;
	Tensor *q_, *k_;
};
