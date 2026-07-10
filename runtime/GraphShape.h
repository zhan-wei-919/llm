#pragma once
#include <cstddef>

struct GraphShape {
	int batch;
	int total_tokens;

	bool operator==(const GraphShape &other) const {
		return  batch == other.batch && total_tokens == other.total_tokens;
	}
};

// 哈希值 = [ batch 的 32 位 ][ total_tokens 的 32 位 ]
// 比如{8, 2048} = 0x00000008_00000800
// 不管 lens 是 [1,1,62] 还是 [16,16,32]，都使用同一张图
struct GraphShapeHash {
	size_t operator() (const GraphShape &shape) const {
		return
			static_cast<size_t>(shape.batch) << 32 |
			static_cast<unsigned>(shape.total_tokens);
	}
};
