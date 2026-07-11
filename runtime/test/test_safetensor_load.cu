#include "../LLM.cuh"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main() {
	const std::string path = "/tmp/test_safetensor_load.safetensors";
	const std::string header = R"({
  "direct": {"dtype":"F32","shape":[3],"data_offsets":[0,12]},
  "matrix": {"dtype":"F32","shape":[2,3],"data_offsets":[12,36]},
  "half_matrix": {"dtype":"BF16","shape":[2,2],"data_offsets":[36,44]}
})";
	const float direct[3] = {7.0f, 8.0f, 9.0f};
	const float matrix[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
	const uint16_t half_matrix[4] = {10, 11, 12, 13};

	std::ofstream file(path, std::ios::binary);
	uint64_t header_size = header.size();
	for (int i = 0; i < 8; ++i) file.put(static_cast<char>((header_size >> (8 * i)) & 0xff));
	file.write(header.data(), static_cast<std::streamsize>(header.size()));
	file.write(reinterpret_cast<const char *>(direct), sizeof(direct));
	file.write(reinterpret_cast<const char *>(matrix), sizeof(matrix));
	file.write(reinterpret_cast<const char *>(half_matrix), sizeof(half_matrix));
	file.close();

	LLM llm(/*max_tensors=*/3);
	Tensor *d_direct = llm.parameter("direct", {3}, Dtype::F32);
	Tensor *d_matrix = llm.parameter("matrix", {3, 2}, Dtype::F32);
	Tensor *d_half = llm.parameter("half_matrix", {2, 2}, Dtype::BF16);
	llm.finalize();

	std::vector<std::string> need_transpose = {"matrix", "half_matrix"};
	llm.load_safetensor(path, [&](const std::string &name) {
		return std::find(need_transpose.begin(), need_transpose.end(), name) != need_transpose.end();
	});

	float got_direct[3], got_matrix[6];
	uint16_t got_half[4];
	cudaMemcpy(got_direct, d_direct->ptr, sizeof(got_direct), cudaMemcpyDeviceToHost);
	cudaMemcpy(got_matrix, d_matrix->ptr, sizeof(got_matrix), cudaMemcpyDeviceToHost);
	cudaMemcpy(got_half, d_half->ptr, sizeof(got_half), cudaMemcpyDeviceToHost);

	const float expected_matrix[6] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
	const uint16_t expected_half[4] = {10, 12, 11, 13};
	for (int i = 0; i < 3; ++i) assert(got_direct[i] == direct[i]);
	for (int i = 0; i < 6; ++i) assert(got_matrix[i] == expected_matrix[i]);
	for (int i = 0; i < 4; ++i) assert(got_half[i] == expected_half[i]);

	std::remove(path.c_str());
	std::printf("test_safetensor_load PASS\n");
	return 0;
}
