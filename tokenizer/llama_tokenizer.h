#pragma once
#include "../utils/json_parser.h"
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace llama_tokenizer_detail {

struct Data {
	std::unordered_map<std::string, int> vocab;
	std::vector<std::string> id_to_token;
	std::unordered_map<std::string, int> merge_rank;
};

inline std::string pair_key(const std::string &left, const std::string &right) {
	return left + '\0' + right;
}

inline void parse_vocab(json::Cursor &json, Data &data) {
	json.expect('{');
	if (!json.consume('}')) {
		do {
			std::string token = json.string();
			json.expect(':');
			int id = static_cast<int>(json.uint64());
			data.vocab.emplace(token, id);
			if ((int)data.id_to_token.size() <= id) data.id_to_token.resize(id + 1);
			data.id_to_token[id] = std::move(token);
		} while (json.consume(','));
		json.expect('}');
	}
}

inline void parse_merges(json::Cursor &json, Data &data) {
	json.expect('[');
	int rank = 0;
	if (!json.consume(']')) {
		do {
			std::string merge = json.string();
			size_t split = merge.find(' ');
			data.merge_rank.emplace(pair_key(merge.substr(0, split), merge.substr(split + 1)), rank++);
		} while (json.consume(','));
		json.expect(']');
	}
}

inline void parse_model(json::Cursor &json, Data &data) {
	json.expect('{');
	if (!json.consume('}')) {
		do {
			std::string key = json.string();
			json.expect(':');
			if (key == "vocab") parse_vocab(json, data);
			else if (key == "merges") parse_merges(json, data);
			else json.skip_value();
		} while (json.consume(','));
		json.expect('}');
	}
}

inline Data load(const std::string &path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	std::string text(static_cast<size_t>(file.tellg()), '\0');
	file.seekg(0);
	file.read(text.data(), static_cast<std::streamsize>(text.size()));
	json::Cursor json(text);
	Data data;
	json.expect('{');
	if (!json.consume('}')) {
		do {
			std::string key = json.string();
			json.expect(':');
			if (key == "model") parse_model(json, data);
			else json.skip_value();
		} while (json.consume(','));
		json.expect('}');
	}
	return data;
}

inline const Data &get(const std::string &path) {
	static std::unordered_map<std::string, Data> cache;
	auto it = cache.find(path);
	if (it == cache.end()) it = cache.emplace(path, load(path)).first;
	return it->second;
}

inline std::vector<std::string> utf8_characters(const std::string &text) {
	std::vector<std::string> result;
	for (size_t i = 0; i < text.size();) {
		unsigned char c = static_cast<unsigned char>(text[i]);
		size_t bytes = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
		result.emplace_back(text.substr(i, bytes));
		i += bytes;
	}
	return result;
}

inline std::string normalize(const std::string &text) {
	const std::string marker = "\xE2\x96\x81"; // U+2581 LOWER ONE EIGHTH BLOCK
	std::string result = marker;
	for (char c : text) {
		if (c == ' ') result += marker;
		else result.push_back(c);
	}
	return result;
}

inline void encode_piece(const std::string &text, const Data &data, std::vector<int> &ids) {
	if (text.empty()) return;
	std::vector<std::string> symbols = utf8_characters(normalize(text));
	while (symbols.size() > 1) {
		int best_rank = std::numeric_limits<int>::max();
		std::string best_left, best_right;
		for (size_t i = 0; i + 1 < symbols.size(); ++i) {
			auto it = data.merge_rank.find(pair_key(symbols[i], symbols[i + 1]));
			if (it != data.merge_rank.end() && it->second < best_rank) {
				best_rank = it->second;
				best_left = symbols[i];
				best_right = symbols[i + 1];
			}
		}
		if (best_rank == std::numeric_limits<int>::max()) break;
		std::vector<std::string> merged;
		for (size_t i = 0; i < symbols.size();) {
			if (i + 1 < symbols.size() && symbols[i] == best_left && symbols[i + 1] == best_right) {
				merged.push_back(symbols[i] + symbols[i + 1]);
				i += 2;
			} else {
				merged.push_back(std::move(symbols[i++]));
			}
		}
		symbols = std::move(merged);
	}

	const char hex[] = "0123456789ABCDEF";
	for (const std::string &symbol : symbols) {
		auto found = data.vocab.find(symbol);
		if (found != data.vocab.end()) {
			ids.push_back(found->second);
			continue;
		}
		for (unsigned char byte : symbol) {
			std::string fallback = "<0x00>";
			fallback[3] = hex[byte >> 4];
			fallback[4] = hex[byte & 15];
			ids.push_back(data.vocab.at(fallback));
		}
	}
}

inline void replace_marker(std::string &text) {
	const std::string marker = "\xE2\x96\x81";
	for (size_t pos = 0; (pos = text.find(marker, pos)) != std::string::npos;) {
		text.replace(pos, marker.size(), " ");
		++pos;
	}
}

} // namespace llama_tokenizer_detail

inline std::vector<int> tokenize(const std::string &text, const std::string &tokenizer_json) {
	const auto &data = llama_tokenizer_detail::get(tokenizer_json);
	std::vector<int> ids = {data.vocab.at("<s>")};
	const std::string special[] = {"<unk>", "<s>", "</s>"};
	for (size_t begin = 0; begin < text.size();) {
		size_t next = std::string::npos;
		const std::string *matched = nullptr;
		for (const std::string &token : special) {
			size_t pos = text.find(token, begin);
			if (pos < next) { next = pos; matched = &token; }
		}
		if (!matched) {
			llama_tokenizer_detail::encode_piece(text.substr(begin), data, ids);
			break;
		}
		llama_tokenizer_detail::encode_piece(text.substr(begin, next - begin), data, ids);
		ids.push_back(data.vocab.at(*matched));
		begin = next + matched->size();
	}
	return ids;
}

inline std::string detokenize(const std::vector<int> &ids, const std::string &tokenizer_json) {
	const auto &data = llama_tokenizer_detail::get(tokenizer_json);
	std::string text;
	for (int id : ids) {
		if (id == data.vocab.at("<unk>") || id == data.vocab.at("<s>") || id == data.vocab.at("</s>")) continue;
		const std::string &token = data.id_to_token[id];
		if (token.size() == 6 && token.compare(0, 3, "<0x") == 0 && token[5] == '>') {
			auto hex = [](char c) { return c <= '9' ? c - '0' : c - 'A' + 10; };
			text.push_back(static_cast<char>((hex(token[3]) << 4) | hex(token[4])));
		} else {
			std::string piece = token;
			llama_tokenizer_detail::replace_marker(piece);
			text += piece;
		}
	}
	if (!text.empty() && text[0] == ' ') text.erase(text.begin());
	return text;
}
