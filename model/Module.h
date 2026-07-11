#pragma once
#include "../runtime/LLM.cuh"

class Module {
public:

protected:
	template<typename Derived>
	static void attach(LLM &llm, Derived &module) {
		llm.append(module);
	}
};
