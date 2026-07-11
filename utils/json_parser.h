#pragma once
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace json {

// 轻量级 JSON 游标：按需读取字段，不构建整棵 DOM。
class Cursor {
public:
	explicit Cursor(const std::string &text) : text_(text) {}

	void whitespace() {
		while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
	}

	bool consume(char c) {
		whitespace();
		if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; }
		return false;
	}

	void expect(char c) {
		if (!consume(c)) fail(std::string("expected '") + c + "'");
	}

	std::string string() {
		whitespace();
		if (pos_ >= text_.size() || text_[pos_++] != '"') fail("expected string");
		std::string out;
		while (pos_ < text_.size()) {
			char c = text_[pos_++];
			if (c == '"') return out;
			if (c != '\\') { out.push_back(c); continue; }
			if (pos_ >= text_.size()) fail("unfinished escape");
			char e = text_[pos_++];
			switch (e) {
				case '"': case '\\': case '/': out.push_back(e); break;
				case 'b': out.push_back('\b'); break;
				case 'f': out.push_back('\f'); break;
				case 'n': out.push_back('\n'); break;
				case 'r': out.push_back('\r'); break;
				case 't': out.push_back('\t'); break;
				default: fail("unsupported string escape");
			}
		}
		fail("unterminated string");
	}

	uint64_t uint64() {
		whitespace();
		if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_])))
			fail("expected unsigned integer");
		uint64_t value = 0;
		while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
			unsigned digit = static_cast<unsigned>(text_[pos_++] - '0');
			if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) fail("integer overflow");
			value = value * 10 + digit;
		}
		return value;
	}

	std::vector<int> int_array() {
		std::vector<int> result;
		expect('[');
		if (consume(']')) return result;
		do {
			uint64_t value = uint64();
			if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) fail("integer is too large");
			result.push_back(static_cast<int>(value));
		} while (consume(','));
		expect(']');
		return result;
	}

	std::vector<uint64_t> uint64_array() {
		std::vector<uint64_t> result;
		expect('[');
		if (!consume(']')) {
			do result.push_back(uint64()); while (consume(','));
			expect(']');
		}
		return result;
	}

	void skip_value() {
		whitespace();
		if (pos_ >= text_.size()) fail("expected value");
		if (text_[pos_] == '"') { string(); return; }
		if (text_[pos_] == '{') {
			expect('{');
			if (consume('}')) return;
			do { string(); expect(':'); skip_value(); } while (consume(','));
			expect('}');
			return;
		}
		if (text_[pos_] == '[') {
			expect('[');
			if (consume(']')) return;
			do skip_value(); while (consume(','));
			expect(']');
			return;
		}
		while (pos_ < text_.size() && text_[pos_] != ',' && text_[pos_] != '}' && text_[pos_] != ']'
		       && !std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
	}

	[[noreturn]] void fail(const std::string &message) const {
		throw std::runtime_error("invalid JSON at byte " + std::to_string(pos_) + ": " + message);
	}

private:
	const std::string &text_;
	size_t pos_ = 0;
};

} // namespace json
