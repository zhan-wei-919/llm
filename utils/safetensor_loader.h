#pragma once
#include "../core/Dtype.h"
#include "json_parser.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct layer_info {
	Dtype d;
	std::vector<int> shape;
	uint64_t start; // 相对 raw tensor data 起点的偏移
	uint64_t end;
};

using safetensor_infos = std::unordered_map<std::string, layer_info>;

namespace safetensor_detail {

inline Dtype parse_dtype(const std::string &name) {
	if (name == "F32") return Dtype::F32;
	if (name == "F16") return Dtype::F16;
	if (name == "BF16") return Dtype::BF16;
	throw std::runtime_error("unsupported safetensors dtype: " + name);
}

inline layer_info parse_layer(json::Cursor &json) {
	json.expect('{');
	std::string dtype;
	std::vector<int> shape;
	std::vector<uint64_t> offsets;
	bool has_dtype = false, has_shape = false, has_offsets = false;
	if (!json.consume('}')) {
		do {
			std::string key = json.string();
			json.expect(':');
			if (key == "dtype") { dtype = json.string(); has_dtype = true; }
			else if (key == "shape") { shape = json.int_array(); has_shape = true; }
			else if (key == "data_offsets") { offsets = json.uint64_array(); has_offsets = true; }
			else json.skip_value();
		} while (json.consume(','));
		json.expect('}');
	}
	if (!has_dtype || !has_shape || !has_offsets || offsets.size() != 2)
		throw std::runtime_error("tensor entry must contain dtype, shape and two data_offsets");
	if (offsets[0] > offsets[1]) throw std::runtime_error("tensor data_offsets are reversed");
	return {parse_dtype(dtype), std::move(shape), offsets[0], offsets[1]};
}

inline uint64_t read_header_size(std::ifstream &file) {
	unsigned char bytes[8];
	file.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
	if (!file) throw std::runtime_error("failed to read safetensors header size");
	uint64_t size = 0;
	for (int i = 0; i < 8; ++i) size |= static_cast<uint64_t>(bytes[i]) << (8 * i);
	return size;
}

} // namespace safetensor_detail

inline safetensor_infos weight_loader(const std::string &path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) throw std::runtime_error("failed to open safetensors file: " + path);
	uint64_t file_size = static_cast<uint64_t>(file.tellg());
	if (file_size < 8) throw std::runtime_error("safetensors file is smaller than its 8-byte prefix");
	file.seekg(0);
	uint64_t header_size = safetensor_detail::read_header_size(file);
	if (header_size > file_size - 8) throw std::runtime_error("safetensors header exceeds file size");

	std::string header(static_cast<size_t>(header_size), '\0');
	file.read(header.data(), static_cast<std::streamsize>(header_size));
	if (!file) throw std::runtime_error("failed to read safetensors JSON header");

	json::Cursor json(header);
	safetensor_infos infos;
	json.expect('{');
	if (!json.consume('}')) {
		do {
			std::string name = json.string();
			json.expect(':');
			if (name == "__metadata__") json.skip_value();
			else infos.emplace(std::move(name), safetensor_detail::parse_layer(json));
		} while (json.consume(','));
		json.expect('}');
	}

	uint64_t data_size = file_size - 8 - header_size;
	for (const auto &entry : infos)
		if (entry.second.end > data_size)
			throw std::runtime_error("tensor " + entry.first + " exceeds safetensors data section");
	return infos;
}
