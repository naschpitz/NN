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

      //-- std::ostream rendering (non-TUI mode) --//
      void renderLiveTable(std::ostream& out);
      void clearLiveTable(std::ostream& out);
      void renderFinalSummary(std::ostream& out);

      //-- TUI table lines --//
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
          // GPU-profiled sub-phase breakdown (accumulated per step)
          double gpuProfileCnnFwd = 0.0;
          double gpuProfileAnnFwd = 0.0;
          double gpuProfileAnnBwd = 0.0;
          double gpuProfileCnnBwd = 0.0;
          double gpuProfileCnnAcc = 0.0;
          double gpuProfileAnnAcc = 0.0;
          double gpuProfileLoss = 0.0;
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

      //-- GPU profile hot state (lock-free, per-row) --//
      struct GpuProfileRow {
          double cnnFwd = 0.0;
          double annFwd = 0.0;
          double annBwd = 0.0;
          double cnnBwd = 0.0;
          double cnnAcc = 0.0;
          double annAcc = 0.0;
          double loss = 0.0;
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

      //-- GPU profile aggregates --//
      double epochGpuProfileCnnFwd = 0.0;
      double epochGpuProfileAnnFwd = 0.0;
      double epochGpuProfileAnnBwd = 0.0;
      double epochGpuProfileCnnBwd = 0.0;
      double epochGpuProfileCnnAcc = 0.0;
      double epochGpuProfileAnnAcc = 0.0;
      double epochGpuProfileLoss = 0.0;
      ulong epochGpuProfileKernelCalls = 0;
      double totalGpuProfileCnnFwd = 0.0;
      double totalGpuProfileAnnFwd = 0.0;
      double totalGpuProfileAnnBwd = 0.0;
      double totalGpuProfileCnnBwd = 0.0;
      double totalGpuProfileCnnAcc = 0.0;
      double totalGpuProfileAnnAcc = 0.0;
      double totalGpuProfileLoss = 0.0;
      ulong totalGpuProfileKernelCalls = 0;

      //-- Published snapshot for rendering --//
      mutable std::mutex mutex;
      StepView lastStep;
      int lastRenderedLines = 0;
      ulong lastRenderedBatchNumber = static_cast<ulong>(-1);

      //-- Helpers --//
      static const char* phaseLabel(CNN::TimingPhase phase);
      static std::string fmt(double v, int width, int prec = 1);
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGPROFILER_HPP
