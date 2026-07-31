#pragma once
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <deque>
#include <functional>
#include <utility>
#include "Scheduler.h"

class PipelineDriver {
public:
	enum class Pump {LAUNCHED, DRAINED, IDLE};
	using LaunchFn = std::function<void(const ScheduledBatch &, int buffer_slot)>;
	using FinishedFn = std::function<void(std::vector<Request> &&)>;

	struct StepTime {double sched_us = 0, wait_us = 0, update_us = 0;};

	PipelineDriver(Scheduler &sched, const int *h_tok, int tok_stride, cudaStream_t stream, LaunchFn launch, FinishedFn on_finished)
	: sched_(sched), h_tok_(h_tok), tok_stride_(tok_stride), stream_(stream), launch_(std::move(launch)), on_finished_(std::move(on_finished)) {
		for (Slot &slot : slots_) cudaEventCreate(&slot.ready);
	}

	~PipelineDriver() {
		drain();
		for (Slot &slot : slots_) cudaEventDestroy(slot.ready);
	}

	Pump pump() {
		times_ = {};
		bool launched = false;
		for (ExecutionPhase phase : {ExecutionPhase::PREFILL, ExecutionPhase::DECODE}) {
			if (inflight_.size() == slots_.size()) consume_oldest();
			auto c0 = now();
			ScheduledBatch batch = sched_.schedule(phase);
			times_.sched_us += us_since(c0);
			if (batch.empty()) continue;
			int slot = free_slot();
			launch_(batch, slot);
			cudaEventRecord(slots_[slot].ready, stream_);
			slots_[slot].batch = std::move(batch);
			slots_[slot].occupied = true;
			inflight_.push_back(slot);
			launched = true;
		}
		if (launched) return Pump::LAUNCHED;
		if (!inflight_.empty()) {
			consume_oldest();
			return Pump::DRAINED;
		}
		sched_.flush_release();
		return sched_.num_requests() == 0 ? Pump::IDLE : Pump::DRAINED;
	}

	void drain() {
		while (!inflight_.empty()) consume_oldest();
		sched_.flush_release();
	}

	void run_to_idle() {while (pump() != Pump::IDLE) {}}
	const StepTime &last_times() const {return times_;}

private:
	struct Slot {
		cudaEvent_t ready;
		ScheduledBatch batch;
		bool occupied = false;
	};

	void consume_oldest() {
		int slot_id = inflight_.front();
		inflight_.pop_front();
		Slot &slot = slots_[slot_id];
		auto c0 = now();
		cudaEventSynchronize(slot.ready);
		times_.wait_us += us_since(c0);
		auto c1 = now();
		sched_.flush_release();
		const StepPlan &plan = slot.batch.plan;
		on_finished_(sched_.update(slot.batch, std::vector<int>(
			h_tok_ + slot_id * tok_stride_,
			h_tok_ + slot_id * tok_stride_ + plan.req_ids.size())));
		times_.update_us += us_since(c1);
		slot.batch = {};
		slot.occupied = false;
	}

	int free_slot() const {
		for (int i = 0; i < (int)slots_.size(); ++i)
			if (!slots_[i].occupied) return i;
		return -1;
	}

	static std::chrono::steady_clock::time_point now() {return std::chrono::steady_clock::now();}
	static double us_since(std::chrono::steady_clock::time_point t0) {
		return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
	}

	Scheduler		&sched_;
	const int		*h_tok_;
	const int		tok_stride_;
	cudaStream_t		stream_;
	LaunchFn		launch_;
	FinishedFn		on_finished_;
	std::array<Slot, 2>	slots_;
	std::deque<int>		inflight_;
	StepTime		times_;
};
