#include "../Engine.cuh"
#include "../tensor/Arena.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 玩具配置: 6 个 query 头共享 2 个 KV 头 (GROUP=3), 每头 12 维
constexpr int NH = 6, NKV = 2, HS = 12;
constexpr int Q_STRIDE = NH * HS;		// 72
constexpr int KV_STRIDE = NKV * HS;		// 24
constexpr int MAX_SEQS = 5, MAX_SEQ_LEN = 64;
constexpr int T = 3;				// prompt 长度

static float frand() { return (rand() % 200 - 100) / 100.0f; }	// [-1, 1)

int main() {
	// ---------- 装配: Arena → KV_Pool → Engine ----------
	int W = (MAX_SEQ_LEN + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
	Arena a(8);
	// Engine 的 device 工作数组 (内容是 int; Dtype 暂无 I32, 用同为 4 字节的 F32 占位)
	Tensor *t_table = a.alloc({MAX_SEQS, W},            Dtype::F32);
	Tensor *t_len   = a.alloc({MAX_SEQS},               Dtype::F32);
	Tensor *t_pos   = a.alloc({MAX_SEQS},               Dtype::F32);
	Tensor *t_ids   = a.alloc({MAX_SEQS * MAX_SEQ_LEN}, Dtype::F32);
	Tensor *t_cu    = a.alloc({MAX_SEQS + 1},           Dtype::F32);
	a.finalize();
	KVAlloc kva = a.alloc_kv_pool(0.05);	// 测试不用吃满整卡
	KV_Pool pool(kva.k_base, kva.v_base, Dtype::F32, KV_STRIDE, kva.bytes_each,
	             MAX_SEQS, MAX_SEQ_LEN);
	Engine eng(pool, NH, NKV, HS,
	           (int *)t_table->ptr, (int *)t_len->ptr, (int *)t_pos->ptr,
	           (int *)t_ids->ptr, (int *)t_cu->ptr);

	// ---------- 造数据: 一条序列, prompt 3 行 + decode 1 行, 共 4 行 ----------
	srand(42);
	float hq[(T + 1) * Q_STRIDE], hk[(T + 1) * KV_STRIDE], hv[(T + 1) * KV_STRIDE];
	for (int i = 0; i < (T + 1) * Q_STRIDE; ++i)  hq[i] = frand();
	for (int i = 0; i < (T + 1) * KV_STRIDE; ++i) hk[i] = frand();
	for (int i = 0; i < (T + 1) * KV_STRIDE; ++i) hv[i] = frand();

	float *dq, *dk, *dv, *dout;
	cudaMalloc(&dq,   sizeof(float) * (T + 1) * Q_STRIDE);
	cudaMalloc(&dk,   sizeof(float) * (T + 1) * KV_STRIDE);
	cudaMalloc(&dv,   sizeof(float) * (T + 1) * KV_STRIDE);
	cudaMalloc(&dout, sizeof(float) * (T + 1) * Q_STRIDE);
	cudaMemcpy(dq, hq, sizeof(float) * (T + 1) * Q_STRIDE, cudaMemcpyHostToDevice);
	cudaMemcpy(dk, hk, sizeof(float) * (T + 1) * KV_STRIDE, cudaMemcpyHostToDevice);
	cudaMemcpy(dv, hv, sizeof(float) * (T + 1) * KV_STRIDE, cudaMemcpyHostToDevice);

	// ---------- 跑: 前 3 行走 prefill, 第 4 行走 decode ----------
	int slot = eng.alloc_seq();
	int slots[1] = {slot}, lens[1] = {T};
	eng.prefill(slots, 1, lens, dq, dk, dv, dout);
	eng.decode(slots, 1,
	           dq + T * Q_STRIDE, dk + T * KV_STRIDE, dv + T * KV_STRIDE,
	           dout + T * Q_STRIDE);

	float gpu[Q_STRIDE];
	cudaMemcpy(gpu, dout + T * Q_STRIDE, sizeof(float) * Q_STRIDE, cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

	// ---------- CPU 参照: 连续布局直接算 decode 那一步 ----------
	// 新 token (第 3 行) 对全部历史 (0..3 共 4 行) 做 attention.
	// GPU 侧这 4 行 K/V 是 scatter 分两次写进 pool、decode 查页表读出来的;
	// CPU 侧直接在连续数组上算. 两边只应差浮点误差.
	const int LEN = T + 1, GROUP = NH / NKV;
	float ref[Q_STRIDE];
	for (int h = 0; h < NH; ++h) {
		int g = h / GROUP;
		const float *qh = hq + T * Q_STRIDE + h * HS;
		float s[LEN], mx = -INFINITY, z = 0;
		for (int j = 0; j < LEN; ++j) {
			float acc = 0;
			for (int d = 0; d < HS; ++d) acc += qh[d] * hk[j * KV_STRIDE + g * HS + d];
			s[j] = acc / sqrtf((float)HS);
			if (s[j] > mx) mx = s[j];
		}
		for (int j = 0; j < LEN; ++j) { s[j] = expf(s[j] - mx); z += s[j]; }
		for (int d = 0; d < HS; ++d) {
			float acc = 0;
			for (int j = 0; j < LEN; ++j) acc += s[j] / z * hv[j * KV_STRIDE + g * HS + d];
			ref[h * HS + d] = acc;
		}
	}

	float max_diff = 0;
	for (int i = 0; i < Q_STRIDE; ++i)
		max_diff = fmaxf(max_diff, fabsf(gpu[i] - ref[i]));
	printf("decode out[0..3]: gpu = %.4f %.4f %.4f %.4f\n", gpu[0], gpu[1], gpu[2], gpu[3]);
	printf("                  ref = %.4f %.4f %.4f %.4f\n", ref[0], ref[1], ref[2], ref[3]);
	printf("max |gpu - ref| = %g  ->  %s\n", max_diff, max_diff < 1e-4 ? "OK" : "MISMATCH");

	eng.release(slot);
	return max_diff < 1e-4 ? 0 : 1;
}
