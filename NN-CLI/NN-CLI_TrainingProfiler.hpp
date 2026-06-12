#ifndef NN_CLI_TRAININGPROFILER_HPP
#define NN_CLI_TRAININGPROFILER_HPP

#include <CNN_TimingCallback.hpp>

#include <QMutex>

#include <array>
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include <sys/types.h>

namespace NN_CLI
{
  class TrainingProfiler
  {
    public:
      TrainingProfiler();

      void onEvent(CNN::TimingPhase phase, CNN::TimingEvent event, int gpuIndex);
      void onGpuProfile(const std::vector<CNN::GpuPhaseProfile>& profiles, int gpuIndex);
      void setEpoch(ulong epoch);
      void resetRenderState();

      //-- TUI table lines --//
      // maxWidth > 0: exact line width to produce (e.g. the TUI panel's content width);
      // maxWidth == 0: auto-detect the terminal width and keep a stdout safety margin.
      std::vector<std::string> getTimingLines(int maxWidth = 0) const;

      void reset();

    private:
      static constexpr int kNumPhases = static_cast<int>(CNN::TimingPhase::Count);
      static constexpr int kMaxRows = 65;

      using Clock = std::chrono::steady_clock;

      static int rowOf(int gpuIndex)
      {
        return (gpuIndex < 0) ? 0 : (gpuIndex + 1 < kMaxRows ? gpuIndex + 1 : kMaxRows - 1);
      }

      struct StepView {
          std::array<double, kNumPhases> orch{};
          double h2dPerGpu = 0.0;
          double computePerGpu = 0.0;
          double orchTotal = 0.0;
          ulong runs = 0;
          ulong batchNumber = 0;
          bool valid = false;
          // GPU-profiled sub-phase breakdown (accumulated per step), keyed by TimingPhase.
          std::array<double, kNumPhases> gpuProfile{};
          ulong gpuProfileKernelCalls = 0;
      };

      void finalizeStep();

      //-- Configuration --//
      int numGpus = 1;

      //-- Lock-free per-row hot state --//
      Clock::time_point starts[kMaxRows][kNumPhases];
      double stepMs[kMaxRows][kNumPhases];
      ulong stepRuns[kMaxRows];
      bool stepInProgress = false;

      //-- GPU profile hot state (lock-free, per-row), keyed by TimingPhase --//
      struct GpuProfileRow {
          std::array<double, kNumPhases> ms{};
          ulong kernelCalls = 0;
      };

      GpuProfileRow stepGpuProfile[kMaxRows];

      //-- Aggregates --//
      std::array<double, kNumPhases> epochMs{};
      std::array<double, kNumPhases> totalMs{};
      ulong epochRuns = 0;
      ulong totalRuns = 0;
      ulong stepCount = 0;
      ulong totalStepCount = 0;
      ulong currentEpoch = 0;

      //-- GPU profile aggregates (keyed by TimingPhase) --//
      std::array<double, kNumPhases> epochGpuProfile{};
      std::array<double, kNumPhases> totalGpuProfile{};
      ulong epochGpuProfileKernelCalls = 0;
      ulong totalGpuProfileKernelCalls = 0;

      //-- Published snapshot for rendering --//
      mutable QMutex mutex;
      StepView lastStep;
      mutable ulong lastRenderedBatchNumber = static_cast<ulong>(-1);
      mutable int lastRenderedWidth = -1;

      //-- Helpers --//
      static const char* phaseLabel(CNN::TimingPhase phase);
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGPROFILER_HPP
