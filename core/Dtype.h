#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

enum class Dtype {F32, BF16, F16, I32};

constexpr size_t dtype_size(Dtype d) {
	switch (d) {
		case Dtype::F32: return 4;
		case Dtype::BF16: return 2;
		case Dtype::F16: return 2;
		case Dtype::I32: return 4;
	}
	return 0;
}

constexpr const char *dtype_name(Dtype d) {
	switch (d) {
		case Dtype::F32: return "f32";
		case Dtype::BF16: return "bf16";
		case Dtype::F16: return "f16";
		case Dtype::I32: return "i32";
	}
	return "invalid";
}

template<typename T> struct dtype_of;
template<> struct dtype_of<float> {static constexpr Dtype value = Dtype::F32; };
template<> struct dtype_of<__nv_bfloat16> {static constexpr Dtype value = Dtype::BF16; };
template<> struct dtype_of<half> {static constexpr Dtype value = Dtype::F16; };
template<> struct dtype_of<int> {static constexpr Dtype value = Dtype::I32; };

template<typename T> struct TypeTag {using type = T; };

template<typename F>
decltype(auto) dtype_dispatch(Dtype d, F &&f) {
	switch (d) {
		case Dtype::F32: return f(TypeTag<float>{});
		case Dtype::BF16: return f(TypeTag<__nv_bfloat16>{});
		case Dtype::F16: return f(TypeTag<half>{});
	}
	__builtin_trap();
}
