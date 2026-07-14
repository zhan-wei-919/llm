#pragma once
#include "../runtime/LLM.h"

class Module {
public:

protected:
	template<typename Derived>
	static void attach(LLM &llm, Derived &module,
	                   std::initializer_list<Tensor *> inputs,
	                   std::initializer_list<Tensor *> outputs) {
		llm.append(module, inputs, outputs);
	}
};
