#include "../../utils.cuh"
#include "../LayerNorm.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

template <typename T>
__global__ void layernorm_scalar(T *out, float *mean_out, float *rstd_out,
                                 const T *x, const T *gamma, const T *beta,
                                 int C, float eps) {
  int bt = blockIdx.x;
  float sum = 0.0f, sum2 = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float v = static_cast<float>(x[bt * C + c]);
    sum += v;
    sum2 += v * v;
  }
  sum = block_sum(sum);
  sum2 = block_sum(sum2);
  float mean = sum / C;
  float var = sum2 / C - mean * mean;
  float rstd = rsqrtf(var + eps);
  if (threadIdx.x == 0) {
    mean_out[bt] = mean;
    rstd_out[bt] = rstd;
  }
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float v = static_cast<float>(x[bt * C + c]);
    float g = static_cast<float>(gamma[c]);
    float b = static_cast<float>(beta[c]);
    out[bt * C + c] = static_cast<T>((v - mean) * rstd * g + b);
  }
}

template <typename T>
void layernorm_cpu(float *out, float *mean_out, float *rstd_out, const T *x,
                   const T *gamma, const T *beta, int BT, int C, float eps) {
  for (int bt = 0; bt < BT; ++bt) {
    float sum = 0.0f, sum2 = 0.0f;
    for (int c = 0; c < C; ++c) {
      float v = static_cast<float>(x[bt * C + c]);
      sum += v;
      sum2 += v * v;
    }
    float mean = sum / C;
    float var = sum2 / C - mean * mean;
    float rstd = 1.0f / sqrtf(var + eps);
    mean_out[bt] = mean;
    rstd_out[bt] = rstd;
    for (int c = 0; c < C; ++c) {
      float v = static_cast<float>(x[bt * C + c]);
      float g = static_cast<float>(gamma[c]);
      float b = static_cast<float>(beta[c]);
      out[bt * C + c] = (v - mean) * rstd * g + b;
    }
  }
}

struct TestCase {
  const char *label;
  int BT, C;
};

static const TestCase cases[] = {
    {"small", 128, 768},   {"gpt2", 4096, 768}, {"gpt2-lg", 4096, 1280},
    {"llama", 4096, 4096}, {"big", 8192, 4096},
};

static constexpr int BLOCK = 256;
static constexpr float EPS = 1e-5f;

template <typename T> void test_type(const char *type_name) {
  printf("\n===== LayerNorm %s =====\n", type_name);
  printf("  %-8s %6s %5s  %10s %10s  %7s %9s  %s\n", "case", "BT", "C",
         "scalar(us)", "vec(us)", "speedup", "BW(GB/s)", "acc");

  int num = sizeof(cases) / sizeof(cases[0]);
  for (int ci = 0; ci < num; ++ci) {
    int BT = cases[ci].BT, C = cases[ci].C;
    size_t s_x = sizeof(T) * BT * C;
    size_t s_out = sizeof(T) * BT * C;
    size_t s_param = sizeof(T) * C;
    size_t s_stat = sizeof(float) * BT;

    T *h_x = (T *)malloc(s_x);
    T *h_gamma = (T *)malloc(s_param);
    T *h_beta = (T *)malloc(s_param);
    float *h_ref_out = (float *)malloc(sizeof(float) * BT * C);
    float *h_ref_mean = (float *)malloc(s_stat);
    float *h_ref_rstd = (float *)malloc(s_stat);
    T *h_scalar = (T *)malloc(s_out);
    T *h_vec = (T *)malloc(s_out);
    float *h_mean = (float *)malloc(s_stat);
    float *h_rstd = (float *)malloc(s_stat);

    fill_matrix(h_x, BT * C, -1.0f, 1.0f);
    fill_matrix(h_gamma, C, 0.5f, 1.5f);
    fill_matrix(h_beta, C, -0.1f, 0.1f);

    layernorm_cpu(h_ref_out, h_ref_mean, h_ref_rstd, h_x, h_gamma, h_beta, BT,
                  C, EPS);

    T *d_x, *d_out, *d_gamma, *d_beta;
    float *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_x, s_x));
    CUDA_CHECK(cudaMalloc(&d_out, s_out));
    CUDA_CHECK(cudaMalloc(&d_gamma, s_param));
    CUDA_CHECK(cudaMalloc(&d_beta, s_param));
    CUDA_CHECK(cudaMalloc(&d_mean, s_stat));
    CUDA_CHECK(cudaMalloc(&d_rstd, s_stat));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, s_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, s_param, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, s_param, cudaMemcpyHostToDevice));

    // scalar
    CUDA_CHECK(cudaMemset(d_out, 0, s_out));
    float ms_scalar = 0;
    TIME_MS(ms_scalar, 20, 200,
            (layernorm_scalar<T><<<BT, BLOCK>>>(d_out, d_mean, d_rstd, d_x,
                                                d_gamma, d_beta, C, EPS)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_scalar, d_out, s_out, cudaMemcpyDeviceToHost));

    // vectorized
    CUDA_CHECK(cudaMemset(d_out, 0, s_out));
    float ms_vec = 0;
    TIME_MS(ms_vec, 20, 200,
            (layernorm<T><<<BT, BLOCK>>>(d_out, d_mean, d_rstd, d_x, d_gamma,
                                         d_beta, C, EPS)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_vec, d_out, s_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean, d_mean, s_stat, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd, d_rstd, s_stat, cudaMemcpyDeviceToHost));

    // 读x + 读gamma + 读beta + 写out = (2*BT*C + 2*C) *
    // sizeof(T)，C很小时gamma/beta可忽略
    double total_bytes = (2.0 * BT * C + 2.0 * C) * sizeof(T) +
                         2.0 * BT * sizeof(float); // mean + rstd
    double bw = total_bytes / (ms_vec * 1e-3) / 1e9;

    bool ok_s = matricesEqual(h_scalar, h_ref_out, BT * C, 2e-2f);
    bool ok_v = matricesEqual(h_vec, h_ref_out, BT * C, 2e-2f);
    bool ok_mean = matricesEqual(h_mean, h_ref_mean, BT, 1e-3f);
    bool ok_rstd = matricesEqual(h_rstd, h_ref_rstd, BT, 1e-3f);
    float speedup = ms_scalar / ms_vec;

    printf("  %-8s %6d %5d  %10.1f %10.1f  %6.2fx %8.1f  %s/%s m=%s r=%s\n",
           cases[ci].label, BT, C, ms_scalar * 1000, ms_vec * 1000, speedup, bw,
           ok_s ? "PASS" : "FAIL", ok_v ? "PASS" : "FAIL",
           ok_mean ? "ok" : "FAIL", ok_rstd ? "ok" : "FAIL");

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    free(h_x);
    free(h_gamma);
    free(h_beta);
    free(h_ref_out);
    free(h_ref_mean);
    free(h_ref_rstd);
    free(h_scalar);
    free(h_vec);
    free(h_mean);
    free(h_rstd);
  }
}

int main() {
  test_type<float>("f32");
  test_type<__nv_bfloat16>("bf16");
  test_type<half>("f16");
}
