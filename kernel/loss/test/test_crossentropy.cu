#include "../CrossEntropy.cuh"
#include "../../utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

template<typename T>
void crossentropy_cpu(
		float *losses,      // [BT]
		float *probs,       // [BT, V]
		const T *logits,    // [BT, V]
		const int *targets, // [BT]
		int BT, int V
) {
	for (int bt = 0; bt < BT; ++bt) {
		const T *row = logits + (size_t)bt * V;
		float *prob_row = probs + (size_t)bt * V;

		float row_max = -INFINITY;
		for (int v = 0; v < V; ++v) {
			float val = static_cast<float>(row[v]);
			if (val > row_max) row_max = val;
		}

		float sum = 0.0f;
		for (int v = 0; v < V; ++v) {
			float e = expf(static_cast<float>(row[v]) - row_max);
			prob_row[v] = e;
			sum += e;
		}
		float inv_z = 1.0f / sum;
		for (int v = 0; v < V; ++v)
			prob_row[v] *= inv_z;

		int tgt = targets[bt];
		float logit_tgt = static_cast<float>(row[tgt]);
		losses[bt] = row_max + logf(sum) - logit_tgt;
	}
}

struct TestCase { const char *label; int BT, V; };

static const TestCase cases[] = {
	{ "tiny",     16,    256  },
	{ "small",   128,   1024  },
	{ "gpt2",   512,  50257  },
	{ "large",  1024,  50257  },
};

template<typename T>
void test_type(const char *type_name) {
	printf("\n===== CrossEntropy %s =====\n", type_name);
	printf("  %-8s %6s %7s  %10s  %s\n",
			"case", "BT", "V", "time(us)", "accuracy");

	std::mt19937 rng(42);

	int num = sizeof(cases) / sizeof(cases[0]);
	for (int ci = 0; ci < num; ++ci) {
		int BT = cases[ci].BT;
		int V  = cases[ci].V;

		size_t s_logits  = sizeof(T) * BT * V;
		size_t s_probs   = sizeof(T) * BT * V;
		size_t s_losses  = sizeof(float) * BT;
		size_t s_targets = sizeof(int) * BT;

		T     *h_logits   = (T*)malloc(s_logits);
		int   *h_targets  = (int*)malloc(s_targets);
		float *h_ref_loss = (float*)malloc(s_losses);
		float *h_ref_prob = (float*)malloc(sizeof(float) * BT * V);
		T     *h_probs    = (T*)malloc(s_probs);
		float *h_losses   = (float*)malloc(s_losses);

		fill_matrix(h_logits, BT * V, -3.0f, 3.0f);
		std::uniform_int_distribution<int> tgt_dist(0, V - 1);
		for (int i = 0; i < BT; ++i)
			h_targets[i] = tgt_dist(rng);

		crossentropy_cpu(h_ref_loss, h_ref_prob, h_logits, h_targets, BT, V);

		T *d_logits; T *d_probs; float *d_losses; int *d_targets;
		CUDA_CHECK(cudaMalloc(&d_logits,  s_logits));
		CUDA_CHECK(cudaMalloc(&d_probs,   s_probs));
		CUDA_CHECK(cudaMalloc(&d_losses,  s_losses));
		CUDA_CHECK(cudaMalloc(&d_targets, s_targets));
		CUDA_CHECK(cudaMemcpy(d_logits,  h_logits,  s_logits,  cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_targets, h_targets, s_targets, cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMemset(d_probs,  0, s_probs));
		CUDA_CHECK(cudaMemset(d_losses, 0, s_losses));
		float ms = 0;
		TIME_MS(ms, 5, 50,
				launch_crossentropy_forward(d_losses, d_probs, d_logits, d_targets,
						1, BT, V));
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		CUDA_CHECK(cudaMemcpy(h_losses, d_losses, s_losses,  cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_probs,  d_probs,  s_probs,   cudaMemcpyDeviceToHost));

		float tol_loss = (sizeof(T) == 2) ? 5e-2f : 1e-3f;
		float tol_prob = (sizeof(T) == 2) ? 5e-2f : 1e-3f;
		bool ok_loss = matricesEqual(h_losses, h_ref_loss, BT, tol_loss);
		bool ok_prob = matricesEqual(h_probs,  h_ref_prob, BT * V, tol_prob);

		printf("  %-8s %6d %7d  %10.1f  loss=%s prob=%s\n",
				cases[ci].label, BT, V,
				ms * 1000,
				ok_loss ? "PASS" : "FAIL",
				ok_prob ? "PASS" : "FAIL");

		if (!ok_loss || !ok_prob) {
			float max_loss_err = 0, max_prob_err = 0;
			for (int k = 0; k < BT; ++k) {
				float diff = fabsf(h_losses[k] - h_ref_loss[k]);
				if (diff > max_loss_err) max_loss_err = diff;
			}
			for (int k = 0; k < BT * V; ++k) {
				float diff = fabsf(static_cast<float>(h_probs[k]) - h_ref_prob[k]);
				if (diff > max_prob_err) max_prob_err = diff;
			}
			printf("           max_err: loss=%.6f prob=%.6f\n", max_loss_err, max_prob_err);
		}

		cudaFree(d_logits); cudaFree(d_probs);
		cudaFree(d_losses); cudaFree(d_targets);
		free(h_logits); free(h_targets);
		free(h_ref_loss); free(h_ref_prob);
		free(h_probs); free(h_losses);
	}
}

int main() {
	test_type<float>("f32");
	test_type<__nv_bfloat16>("bf16");
	test_type<half>("f16");
}
