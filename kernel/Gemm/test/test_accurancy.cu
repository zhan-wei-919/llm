#include "../Gemm.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

struct TestSize { int M, N, K; };

static const TestSize sizes[] = {
	{ 128,   128,   128   },
	{ 256,   256,   256   },
	{ 512,   512,   512   },
	{ 1024,  1024,  1024  },
	{ 4096,  4096,  4096  },
	{ 12288, 12288, 12288 },
	// 非方阵
	{ 1,     4096,  4096  },
	{ 128,   4096,  1024  },
	{ 4096,  1024,  128   },
	// M/N 不被 BM/BN 整除，但 K 和向量宽度对齐
	{ 1000,  1024,  1024  },
	{ 384,   512,   256   },
	{ 2048,  3072,  1024  },
};

template<typename InT, typename OutT>
struct CublasTypeTraits;

template<> struct CublasTypeTraits<__nv_bfloat16, __nv_bfloat16> {
	static constexpr cudaDataType_t in_type  = CUDA_R_16BF;
	static constexpr cudaDataType_t out_type = CUDA_R_16BF;
};
template<> struct CublasTypeTraits<__nv_bfloat16, float> {
	static constexpr cudaDataType_t in_type  = CUDA_R_16BF;
	static constexpr cudaDataType_t out_type = CUDA_R_32F;
};
template<> struct CublasTypeTraits<half, half> {
	static constexpr cudaDataType_t in_type  = CUDA_R_16F;
	static constexpr cudaDataType_t out_type = CUDA_R_16F;
};
template<> struct CublasTypeTraits<half, float> {
	static constexpr cudaDataType_t in_type  = CUDA_R_16F;
	static constexpr cudaDataType_t out_type = CUDA_R_32F;
};

template<typename InT, typename OutT>
void test_type(const char *name, cublasHandle_t handle) {
	using Cfg = GemmConfig<InT, OutT>;
	using Traits = CublasTypeTraits<InT, OutT>;
	cudaStream_t stream = nullptr;

	printf("\n===== %s =====\n", name);
	printf("%-20s %8s %8s %8s  %10s %10s  %6s  %s\n",
			"size", "M", "N", "K", "ours(ms)", "cublas(ms)", "ratio", "accuracy");

	int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
	for (int si = 0; si < num_sizes; ++si) {
		int M = sizes[si].M, N = sizes[si].N, K = sizes[si].K;

		size_t sA = sizeof(InT)  * M * K;
		size_t sB = sizeof(InT)  * K * N;
		size_t sC = sizeof(OutT) * M * N;

		InT  *a_host = (InT*)malloc(sA);
		InT  *b_host = (InT*)malloc(sB);
		OutT *c_ours = (OutT*)malloc(sC);
		OutT *c_ref  = (OutT*)malloc(sC);

		fill_matrix(a_host, M * K, -1.0f, 1.0f);
		fill_matrix(b_host, K * N, -1.0f, 1.0f);

		InT *a_dev, *b_dev; OutT *c_dev;
		CUDA_CHECK(cudaMalloc(&a_dev, sA));
		CUDA_CHECK(cudaMalloc(&b_dev, sB));
		CUDA_CHECK(cudaMalloc(&c_dev, sC));
		CUDA_CHECK(cudaMemcpy(a_dev, a_host, sA, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(b_dev, b_host, sB, cudaMemcpyHostToDevice));

		// 我们的 kernel
		CUDA_CHECK(cudaMemset(c_dev, 0, sC));
		int warmup = (M >= 4096) ? 5 : 20;
		int iters  = (M >= 4096) ? 20 : 50;
		float ms_ours = 0.0f;
		TIME_MS(ms_ours, warmup, iters,
				launch_Gemm_forward<Cfg>(a_dev, b_dev, c_dev, nullptr, 1.0f, 0.0f, M, N, K, stream));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(c_ours, c_dev, sC, cudaMemcpyDeviceToHost));

		// cuBLAS 参照
		CUDA_CHECK(cudaMemset(c_dev, 0, sC));
		float alpha = 1.0f, beta = 0.0f;
		float ms_cublas = 0.0f;
		TIME_MS(ms_cublas, warmup, iters, cublasGemmEx(handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				N, M, K,
				&alpha,
				b_dev, Traits::in_type, N,
				a_dev, Traits::in_type, K,
				&beta,
				c_dev, Traits::out_type, N,
				CUBLAS_COMPUTE_32F,
				CUBLAS_GEMM_DEFAULT));
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(c_ref, c_dev, sC, cudaMemcpyDeviceToHost));

		bool ok = matricesEqual(c_ours, c_ref, M * N, 2e-2f);
		float ratio = ms_cublas / ms_ours;

		char size_str[64];
		snprintf(size_str, sizeof(size_str), "%dx%dx%d", M, N, K);
		printf("%-20s %8d %8d %8d  %10.3f %10.3f  %5.2fx  %s\n",
				size_str, M, N, K, ms_ours, ms_cublas, ratio, ok ? "PASS" : "FAIL");

		cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
		free(a_host); free(b_host); free(c_ours); free(c_ref);
	}
}

int main() {
	cublasHandle_t handle;
	cublasCreate(&handle);

	test_type<__nv_bfloat16, __nv_bfloat16>("bf16 -> bf16", handle);
	test_type<__nv_bfloat16, float>         ("bf16 -> f32",  handle);
	test_type<half, half>                   ("f16  -> f16",  handle);
	test_type<half, float>                  ("f16  -> f32",  handle);

	cublasDestroy(handle);
}
