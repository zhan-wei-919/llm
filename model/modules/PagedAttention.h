#pragma once
#include <string>
#include "../Module.h"
#include "../../tensor/Tensor.h"
#include "../../runtime/Engine.h"

template<typename T>
class PagedAttention : public Module {
public:
	PagedAttention(LLM &llm, Engine &engine, Tensor *q, Tensor *k, Tensor *v, int layer)
	: engine_(engine), q_(q), k_(k), v_(v), layer_(layer) {
		out_ = llm.arena().alloc({q_->shape[0], q_->shape[1]}, dtype_of<T>::value);
		attach(llm, *this, {q_, k_, v_}, {out_});
	}

	void forward(const GraphShape &, cudaStream_t stream) {
		const T *q = static_cast<const T*>(q_->ptr);
		const T *k = static_cast<const T*>(k_->ptr);
		const T *v = static_cast<const T*>(v_->ptr);
		T *out = static_cast<T*>(out_->ptr);
		engine_.forward_layer(layer_, q, k, v, out, stream);
	}

	Tensor *out() {return out_;}

private:
	Engine &engine_;
	Tensor *q_, *k_, *v_, *out_;
	int layer_;
};
