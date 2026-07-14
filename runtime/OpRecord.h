#pragma once
#include <cuda_runtime.h>
#include <initializer_list>
#include <vector>
#include "../tensor/Tensor.h"
#include "GraphShape.h"

class OpRecord {
public:
	template<typename Module>
	OpRecord(Module &module, std::initializer_list<Tensor *> inputs,
	         std::initializer_list<Tensor *> outputs)
	: module_(&module), forward_(&call_forward<Module>)
	, inputs_(inputs), outputs_(outputs) {}

	void forward(const GraphShape &shape, cudaStream_t stream) const {forward_(module_, shape, stream);}
	Tensor *input(size_t index = 0) const { return inputs_[index]; }
	Tensor *output(size_t index = 0) const { return outputs_[index]; }

private:
	using ForwardFn = void (*)(void*, const GraphShape &, cudaStream_t);

	template<typename Module>
	static void call_forward(void *module,const GraphShape &shape, cudaStream_t stream) {
		static_cast<Module*>(module)->forward(shape, stream);
	}

	void *module_;
	ForwardFn forward_;
	std::vector<Tensor *> inputs_;
	std::vector<Tensor *> outputs_;
};
