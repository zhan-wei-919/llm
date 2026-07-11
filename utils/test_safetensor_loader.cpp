#include "safetensor_loader.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

int main() {
	const std::string path = "/tmp/test_safetensor_loader.safetensors";
	const std::string header = R"({
  "layer.weight": {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},
  "layer.bias": {"dtype":"F32","shape":[2],"data_offsets":[24,32]},
  "__metadata__": {"format":"pt"}
})";

	std::ofstream file(path, std::ios::binary);
	uint64_t size = header.size();
	for (int i = 0; i < 8; ++i) file.put(static_cast<char>((size >> (8 * i)) & 0xff));
	file.write(header.data(), static_cast<std::streamsize>(header.size()));
	const char data[32] = {};
	file.write(data, sizeof(data));
	file.close();

	auto infos = weight_loader(path);
	assert(infos.size() == 2);
	const auto &weight = infos.at("layer.weight");
	assert(weight.d == Dtype::F32);
	assert((weight.shape == std::vector<int>{2, 3}));
	assert(weight.start == 0 && weight.end == 24);
	const auto &bias = infos.at("layer.bias");
	assert((bias.shape == std::vector<int>{2}));
	assert(bias.start == 24 && bias.end == 32);
	std::remove(path.c_str());
	std::printf("test_safetensor_loader PASS\n");
	return 0;
}
