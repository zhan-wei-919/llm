#pragma once
#include <cuda_bf16.h>     // __nv_bfloat16
#include <cuda_fp16.h>     // half
#include <vector_types.h>  // float4

template<typename InT> struct MmaTraits;
template<> struct MmaTraits<__nv_bfloat16>		{ using Acc = float; };
template<> struct MmaTraits<half>				{ using Acc = float; };
template<> struct MmaTraits<signed char>		{ using Acc = int;   };

template<typename InElem, typename OutElem>
struct GemmConfig {
		static constexpr int BM	= 128;
		static constexpr int BN	= 128;
		static constexpr int BK	= 32;
		static constexpr int WM	= 64;
		static constexpr int WN	= 64;
		static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
		static constexpr int NWARPS = WARPS_M * WARPS_N;
		static constexpr int THREADS = 32 * NWARPS;
		static constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
		static constexpr int FM = WM / WMMA_M, FN = WN / WMMA_N;
		
		static constexpr int TM = 8;
		static constexpr int TN = 8;

		using In = InElem;
		using Out = OutElem;
		using Acc = typename MmaTraits<InElem>::Acc;

		static constexpr int VEC = sizeof(float4) / sizeof(In);
		static constexpr int A_F4 = BM * BK / VEC, A_LPT = (A_F4 + THREADS - 1) / THREADS;
		static constexpr int B_F4 = BK * BN / VEC, B_LPT = (B_F4 + THREADS - 1) / THREADS;
		static constexpr int PAD = 8;
		
		static constexpr int SMEM_BYTES = sizeof(InElem) * 2 * BM * (BK + PAD) + sizeof(InElem) * 2 * BK * (BN + PAD);
};
