#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <chrono>
#include "Scheduler.h"

class PipelineDriver {
public:
	enum class Pump {LAUNCHED, DRAINED, IDLE};
	using LaunchFn = std::function<void(const StepPlan &, int parity)>;
	using FinishedFn = std::function<void(std::vector<Request> &&)>;

	struct StepTime {double sched_us = 0, wait_us = 0, update_us = 0;};

	PipelineDriver(Scheduler &sched, const int *h_tok, int tok_stride, cudaStream_t t, LaunchFn launch, FinishedFn on_finished)
	: sched_(sched), h_tok_(h_tok), tok_stride_(tok_stride)
	, t_(t), launch_(std::move(launch)), on_finished_(std::move(on_finished)) {
		cudaEventCreate(&ev_tok_[0]);
		cudaEventCreate(&ev_tok_[1]);
	}

	~PipelineDriver() {
		drain();
		cudaEventDestroy(ev_tok_[0]);
		cudaEventDestroy(ev_tok_[1]);
	}

	Pump pump() {
		auto c0 = now();
		StepPlan plan = sched_.schedule(inflight_.empty());
		times_.sched_us = us_since(c0);
		if (plan.empty()) {
			if (!inflight_.empty()) {drain(); return Pump::DRAINED;}
			return Pump::IDLE;
		}
		int p = calls_++ & 1;
		launch_(plan, p);
		cudaEventRecord(ev_tok_[p], t_);
		if (!inflight_.empty()) consume(1 - p);
		inflight_ = std::move(plan);
		return Pump::LAUNCHED;
	}

	void drain(){
		if (inflight_.empty()) return;
		consume((calls_ - 1) & 1);
		inflight_ = StepPlan{};
		sched_.flush_release();
	}

	void run_to_idle() {while (pump() != Pump::IDLE) {}}
	const StepTime &last_times() const {return  times_;}


private:
	void consume(int q) {
		auto c0 = now();
		cudaEventSynchronize(ev_tok_[q]);
		times_.wait_us = us_since(c0);
		auto c1 = now();
		sched_.flush_release();
		on_finished_(sched_.update(inflight_, std::vector<int>(
			h_tok_ + q * tok_stride_,
			h_tok_ + q * tok_stride_ + inflight_.req_ids.size())));
		times_.update_us = us_since(c1);
	}

	static std::chrono::steady_clock::time_point now() {return std::chrono::steady_clock::now();}
	static double us_since(std::chrono::steady_clock::time_point t0) {
		return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
	}

	Scheduler	&sched_;
	const int	*h_tok_;
	const int	tok_stride_;
	cudaStream_t	t_;
	LaunchFn	launch_;
	FinishedFn	on_finished_;
	cudaEvent_t	ev_tok_[2];
	StepPlan	inflight_;
	int		calls_ = 0;
	StepTime	times_;
};
