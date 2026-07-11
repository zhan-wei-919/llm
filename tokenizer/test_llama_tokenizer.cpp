#include "llama_tokenizer.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
	const std::string path = argv[1];
	const std::string cases[] = {
		"Hello world!",
		"TinyLlama is small, but useful.",
		"multiple   spaces\nand a newline",
		"UTF-8: 你好, café, 🚀"
	};
	for (const std::string &text : cases) {
		std::vector<int> ids = tokenize(text, path);
		assert(!ids.empty() && ids[0] == 1);
		std::string decoded = detokenize(ids, path);
		if (decoded != text)
			std::fprintf(stderr, "round-trip mismatch:\nwant=[%s]\ngot =[%s]\n", text.c_str(), decoded.c_str());
		assert(decoded == text);
	}
	assert((tokenize("Hello world!", path) == std::vector<int>{1, 15043, 3186, 29991}));
	assert((tokenize("UTF-8: 你好, café, 🚀", path) == std::vector<int>{
		1, 18351, 29899, 29947, 29901, 29871, 30919, 31076,
		29892, 274, 28059, 29892, 29871, 243, 162, 157, 131
	}));
	std::vector<int> chat = tokenize("<|user|>\nHello</s>\n<|assistant|>", path);
	assert(std::find(chat.begin(), chat.end(), 2) != chat.end());
	assert(detokenize(chat, path).find("</s>") == std::string::npos);
	std::printf("test_llama_tokenizer PASS\n");
	return 0;
}
