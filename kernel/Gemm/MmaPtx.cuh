#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// PTX 层的 ldmatrix / mma.sync 封装。
// 这些指令直接替代 wmma::load_matrix_sync / wmma::mma_sync：
//   - ldmatrix 一条指令为整个 warp 从 shared memory 加载一个 fragment，
//     取代 wmma 展开出的一堆 LDS，缓解 LSU 压力。
//   - mma.sync 直接在寄存器上做 16x8x16 的矩阵乘加，没有 API 包装层。
// 仅支持 sm_80+（mma.m16n8k16 需要 Ampere 起）。

// ldmatrix 的地址操作数必须是 shared 窗口地址，用 cvta 把通用指针转过去。
__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
		return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// 加载 4 个 8x8 的 b16 矩阵（一个 16x16 区域），结果按 mma 的 A 操作数布局摊到各 lane。
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
		asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
				: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
				: "r"(addr));
}

// 带转置的 ldmatrix：B 在 shared memory 里是 [K][N] 行主序，转置后才符合 mma 的 B 操作数布局。
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
		asm volatile(
				"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
				: "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
				: "r"(addr));
}

// D[16x8] += A[16x16] * B[16x8]，A/B 为 16 位，累加 f32。
// IsBF16 选择 bf16 还是 f16 的指令变体（两者 fragment 布局相同，只是 opcode 不同）。
template<bool IsBF16>
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
		if constexpr (IsBF16) {
				asm volatile(
						"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
						"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
						: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
						: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
						  "r"(b[0]), "r"(b[1]));
		} else {
				asm volatile(
						"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
						"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
						: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
						: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
						  "r"(b[0]), "r"(b[1]));
		}
}
