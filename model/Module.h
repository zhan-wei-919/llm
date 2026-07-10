#pragma once
#include <string>
#include <utility>
#include "../runtime/LLM.cuh"

class Module {
public:

protected:
	template<typename Derived>
	static void attach(LLM &llm, std::string name, Derived &module) {
		llm.append(std::move(name), module);
	}
};
