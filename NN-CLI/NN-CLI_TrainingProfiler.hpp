#ifndef NN_CLI_TRAININGPROFILER_HPP
#define NN_CLI_TRAININGPROFILER_HPP

#include <CNN_TimingCallback.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include <sys/types.h>

//===================================================================================================================//
//-- Reusable training-phase profiler --//
//
// Consumes the begin/end phase events emitted by the CNN library (CNN::TimingCallback)
// and turns them into a live, per-step / per-epoch / whole-run breakdown of where the
// training time goes. This is a real, reusable measurement class — not throwaway code:
// wire any library that emits CNN::TimingPhase events into onEvent() and it just works.
//
// Hot path (onEvent) is lock-free: each phase boundary only timestamps / accumulates
// into a row owned by a single thread (row 0 = orchestrator/main thread; rows 1..N =
// per-GPU worker threads). Aggregation and rendering take a small mutex off the hot path.
//
// A "step" is one mini-batch (one weight update). Step boundaries are detected from the
// DataFetch-Begin event, so no extra wiring is needed. Epoch boundaries are fed in via
// setEpoch() from the existing training callback.
//
// Display semantics:
//   * Orchestrator phases (data_fetch, gpu_train, grad_merge, weight_update,
//     kernel_restore) are sequential and non-overlapping: they sum to the batch wall
//     time and carry the % column.
//   * Worker phases (h2d_upload, gpu_compute) run in parallel across GPUs inside
//     gpu_train; they are normalized to a per-GPU average (sum / numGpus) so they line
//     up with the gpu_train wall time. They are shown indented as a sub-breakdown.
//===================================================================================================================//

namespace NN_CLI
{
  class TrainingProfiler
  {
    public:
      TrainingProfiler();

      // Event sink — wire to CNN::Core::setTimingCallback(). Thread-safe, lock-free.
      void onEvent(CNN::TimingPhase phase, CNN::TimingEvent event, int gpuIndex);

      // Number of GPUs, used to normalize the parallel worker phases. Default 1.
      void setNumGpus(int numGpus);

      // Feed the current epoch (from the training callback). A change rolls up the
      // finished epoch and resets per-epoch accumulators.
      void setEpoch(ulong epoch);

      // True once at least one full step has been measured.
      bool hasData() const;

      // Draw the compact live table just below the progress bar, using ANSI cursor
      // save/restore so the progress bar line is left untouched.
      void renderLiveTable(std::ostream& out);

      // Wipe the floating live-table region (call before committing an epoch line).
      void clearLiveTable(std::ostream& out);

      // One-shot tables for the logger / final report.
      void renderEpochSummary(std::ostream& out, ulong epoch);
      void renderFinalSummary(std::ostream& out);

      void reset();

    private:
      static constexpr int kNumPhases = static_cast<int>(CNN::TimingPhase::Count);
      static constexpr int kMaxRows = 65; // row 0 = orchestrator, rows 1..64 = GPUs

      using Clock = std::chrono::steady_clock;

      static int rowOf(int gpuIndex)
      {
        return (gpuIndex < 0) ? 0 : (gpuIndex + 1 < kMaxRows ? gpuIndex + 1 : kMaxRows - 1);
      }

      // Snapshot of one finalized step, in milliseconds, ready to render.
      struct StepView {
          std::array<double, kNumPhases> orch{}; // orchestrator-phase ms (row 0)
          double h2dPerGpu = 0.0; // per-GPU avg ms
          double computePerGpu = 0.0; // per-GPU avg ms
          double orchTotal = 0.0; // sum of orchestrator phases
          ulong runs = 0; // GPU run() launches this step (all GPUs)
          ulong stepNumber = 0;
          bool valid = false;
      };

      void finalizeStep(); // roll current step into epoch/total + publish snapshot

      //-- Configuration --//
      int numGpus = 1;

      //-- Lock-free per-row hot state --//
      Clock::time_point starts[kMaxRows][kNumPhases];
      double stepMs[kMaxRows][kNumPhases];
      ulong stepRuns[kMaxRows];
      bool stepInProgress = false;

      //-- Aggregates (orchestrator phases store row-0 ms; worker phases store the
      //   summed-across-GPU ms, normalized at render time) --//
      std::array<double, kNumPhases> epochMs{};
      std::array<double, kNumPhases> totalMs{};
      ulong epochRuns = 0;
      ulong totalRuns = 0;
      ulong stepCount = 0; // steps in current epoch
      ulong totalStepCount = 0; // steps over whole run
      ulong currentEpoch = 0;

      //-- Published snapshot for rendering (guarded) --//
      mutable std::mutex mutex;
      StepView lastStep;
      int lastRenderedLines = 0;

      //-- Live-render throttle (the callback fires per-sample; cap redraw rate) --//
      Clock::time_point lastRenderTime{};
      bool haveRenderTime = false;
      double minRenderIntervalSec = 0.15;

      //-- Helpers --//
      static const char* phaseLabel(CNN::TimingPhase phase);
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGPROFILER_HPP
