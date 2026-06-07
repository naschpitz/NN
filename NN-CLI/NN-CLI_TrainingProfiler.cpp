#include "NN-CLI_TrainingProfiler.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <sys/ioctl.h>

#include <unistd.h>

namespace NN_CLI
{
  using Phase = CNN::TimingPhase;
  using Event = CNN::TimingEvent;

  namespace
  {
    constexpr Phase kOrchPhases[] = {Phase::DataFetch, Phase::GpuTrain, Phase::GradMerge, Phase::WeightUpdate,
                                     Phase::KernelRestore};

    // GPU-profiled sub-phases, in display order. Stored/aggregated in arrays keyed by TimingPhase.
    constexpr Phase kGpuSubPhases[] = {Phase::CnnForward,  Phase::AnnForward,    Phase::AnnBackward,
                                       Phase::CnnBackward, Phase::CNNAccumulate, Phase::ANNAccumulate,
                                       Phase::LossCompute};

    template <std::size_t N>
    double gpuSubTotal(const std::array<double, N>& a)
    {
      double sum = 0.0;

      for (Phase ph : kGpuSubPhases)
        sum += a[static_cast<int>(ph)];

      return sum;
    }
  } // namespace

  //===================================================================================================================//

  TrainingProfiler::TrainingProfiler()
  {
    this->reset();
  }

  void TrainingProfiler::reset()
  {
    std::lock_guard<std::mutex> lock(this->mutex);

    std::memset(this->stepMs, 0, sizeof(this->stepMs));
    std::memset(this->stepRuns, 0, sizeof(this->stepRuns));
    this->epochMs.fill(0.0);
    this->totalMs.fill(0.0);
    this->epochRuns = 0;
    this->totalRuns = 0;
    this->stepCount = 0;
    this->totalStepCount = 0;
    this->currentEpoch = 0;
    this->stepInProgress = false;
    this->lastStep = StepView{};
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
    this->numGpus = 1;

    std::memset(this->stepGpuProfile, 0, sizeof(this->stepGpuProfile));
    this->epochGpuProfile.fill(0.0);
    this->totalGpuProfile.fill(0.0);
    this->epochGpuProfileKernelCalls = 0;
    this->totalGpuProfileKernelCalls = 0;
  }

  //===================================================================================================================//
  //-- Hot path: phase boundary events (lock-free) --//
  //===================================================================================================================//

  void TrainingProfiler::onEvent(Phase phase, Event event, int gpuIndex)
  {
    const int row = rowOf(gpuIndex);
    const int p = static_cast<int>(phase);

    if (gpuIndex >= 0 && gpuIndex + 1 > this->numGpus)
      this->numGpus = gpuIndex + 1;

    if (event == Event::Begin) {
      if (phase == Phase::DataFetch && row == 0) {
        if (this->stepInProgress)
          this->finalizeStep();

        this->stepInProgress = true;
      }

      if (phase == Phase::GpuCompute && row > 0)
        this->stepRuns[row]++;

      this->starts[row][p] = Clock::now();
    } else {
      const double ms = std::chrono::duration<double, std::milli>(Clock::now() - this->starts[row][p]).count();
      this->stepMs[row][p] += ms;
    }
  }

  //===================================================================================================================//
  //-- Step finalize --//
  //===================================================================================================================//

  void TrainingProfiler::finalizeStep()
  {
    double h2dSum = 0.0;
    double computeSum = 0.0;
    ulong runs = 0;
    std::array<double, kNumPhases> gpuProfileSum{};
    ulong gpuProfileKernelCallsSum = 0;

    for (int g = 1; g <= this->numGpus && g < kMaxRows; g++) {
      h2dSum += this->stepMs[g][static_cast<int>(Phase::H2DUpload)];
      computeSum += this->stepMs[g][static_cast<int>(Phase::GpuCompute)];
      runs += this->stepRuns[g];

      for (Phase ph : kGpuSubPhases)
        gpuProfileSum[static_cast<int>(ph)] += this->stepGpuProfile[g].ms[static_cast<int>(ph)];

      gpuProfileKernelCallsSum += this->stepGpuProfile[g].kernelCalls;
    }

    StepView view;
    double orchTotal = 0.0;

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      view.orch[p] = this->stepMs[0][p];
      orchTotal += this->stepMs[0][p];
    }

    view.orchTotal = orchTotal;
    view.h2dPerGpu = h2dSum / this->numGpus;
    view.computePerGpu = computeSum / this->numGpus;
    view.runs = runs;
    view.batchNumber = this->stepCount;
    view.valid = true;

    // gpuProfileSum is the total across all GPUs, but view.gpuProfile must be per-GPU
    // to be directly comparable with view.computePerGpu (which is also per-GPU).
    // epoch/total accumulators continue to store the raw sum (divided at display time).
    {
      std::array<double, kNumPhases> perGpu{};

      for (Phase ph : kGpuSubPhases) {
        const int p = static_cast<int>(ph);
        perGpu[p] = gpuProfileSum[p] / this->numGpus;
      }

      view.gpuProfile = perGpu;
    }

    view.gpuProfileKernelCalls = gpuProfileKernelCallsSum / this->numGpus;

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      this->epochMs[p] += this->stepMs[0][p];
      this->totalMs[p] += this->stepMs[0][p];
    }

    const int hp = static_cast<int>(Phase::H2DUpload);
    const int cp = static_cast<int>(Phase::GpuCompute);
    this->epochMs[hp] += h2dSum;
    this->totalMs[hp] += h2dSum;
    this->epochMs[cp] += computeSum;
    this->totalMs[cp] += computeSum;
    this->epochRuns += runs;
    this->totalRuns += runs;
    this->stepCount++;
    this->totalStepCount++;

    std::memset(this->stepMs, 0, sizeof(this->stepMs));
    std::memset(this->stepRuns, 0, sizeof(this->stepRuns));
    std::memset(this->stepGpuProfile, 0, sizeof(this->stepGpuProfile));

    for (Phase ph : kGpuSubPhases) {
      const int p = static_cast<int>(ph);
      this->epochGpuProfile[p] += gpuProfileSum[p];
      this->totalGpuProfile[p] += gpuProfileSum[p];
    }

    this->epochGpuProfileKernelCalls += gpuProfileKernelCallsSum;
    this->totalGpuProfileKernelCalls += gpuProfileKernelCallsSum;

    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->lastStep = view;
    }
  }

  //===================================================================================================================//
  //-- GPU profile data (per-sample, lock-free per-row accumulation) --//
  //===================================================================================================================//

  void TrainingProfiler::onGpuProfile(const std::vector<CNN::GpuPhaseProfile>& profiles, int gpuIndex)
  {
    const int row = rowOf(gpuIndex);

    for (const auto& p : profiles) {
      const int idx = static_cast<int>(p.phase);

      if (idx >= 0 && idx < kNumPhases)
        this->stepGpuProfile[row].ms[idx] += p.gpuMs;

      this->stepGpuProfile[row].kernelCalls += p.kernelCalls;
    }
  }

  void TrainingProfiler::resetRenderState()
  {
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
  }

  //===================================================================================================================//
  //-- Epoch boundary --//
  //===================================================================================================================//

  void TrainingProfiler::setEpoch(ulong epoch)
  {
    if (epoch == this->currentEpoch)
      return;

    this->currentEpoch = epoch;
    this->epochMs.fill(0.0);
    this->epochRuns = 0;
    this->stepCount = 0;
    this->epochGpuProfile.fill(0.0);
    this->epochGpuProfileKernelCalls = 0;

    std::lock_guard<std::mutex> lock(this->mutex);
    this->lastStep.valid = false;
  }

  //===================================================================================================================//
  //-- Phase labels / formatting --//
  //===================================================================================================================//

  const char* TrainingProfiler::phaseLabel(Phase phase)
  {
    switch (phase) {
    case Phase::DataFetch:
      return "data_fetch";
    case Phase::GpuTrain:
      return "gpu_train";
    case Phase::GradMerge:
      return "grad_merge";
    case Phase::WeightUpdate:
      return "weight_update";
    case Phase::KernelRestore:
      return "kernel_restore";
    case Phase::H2DUpload:
      return "h2d_upload";
    case Phase::GpuCompute:
      return "gpu_compute";
    case Phase::Augmentation:
      return "augmentation";
    case Phase::CnnForward:
      return "cnn_forward";
    case Phase::AnnForward:
      return "ann_forward";
    case Phase::AnnBackward:
      return "ann_backward";
    case Phase::CnnBackward:
      return "cnn_backward";
    case Phase::CNNAccumulate:
      return "cnn_accumulate";
    case Phase::ANNAccumulate:
      return "ann_accumulate";
    case Phase::LossCompute:
      return "loss";
    default:
      return "?";
    }
  }

  //===================================================================================================================//
  //-- TUI table lines (for CDK label) --//
  //===================================================================================================================//

  std::vector<std::string> TrainingProfiler::getTimingLines(int maxWidth) const
  {
    StepView v;
    {
      std::lock_guard<std::mutex> lock(this->mutex);

      if (!this->lastStep.valid)
        return {" Timing - waiting for first batch"};

      if (this->lastStep.batchNumber == this->lastRenderedBatchNumber)
        return {};

      v = this->lastStep;
    }

    const_cast<TrainingProfiler*>(this)->lastRenderedBatchNumber = v.batchNumber;

    std::vector<std::string> lines;
    const double total = v.orchTotal > 0.0 ? v.orchTotal : 1.0;
    const ulong launchesPerGpu = v.runs / static_cast<ulong>(std::max(1, this->numGpus));
    const double msPerLaunch = launchesPerGpu > 0 ? v.computePerGpu / static_cast<double>(launchesPerGpu) : 0.0;

    constexpr int phaseW = 20;
    constexpr int pctW = 7;
    constexpr int tableOverhead = phaseW + pctW + 10;
    constexpr int minMsW = 10;

    struct winsize ws;
    ulong termWidth = 0;

    if (maxWidth > 0) {
      termWidth = static_cast<ulong>(maxWidth);
    } else {
      if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        termWidth = static_cast<ulong>(ws.ws_col);

      if (termWidth == 0 && ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        termWidth = static_cast<ulong>(ws.ws_col);
    }

    ulong containerWidth = termWidth > 5 ? termWidth - 5 : 120;
    int msW = static_cast<int>(
      containerWidth > static_cast<ulong>(tableOverhead + minMsW) ? containerWidth - tableOverhead : minMsW);
    msW = std::max(msW, minMsW);

    auto sep = [&]() {
      lines.push_back("+-" + std::string(phaseW, '-') + "-+-" + std::string(msW, '-') + "-+-" + std::string(pctW, '-') +
                      "-+");
    };

    auto row = [&](const std::string& phase, const std::string& ms, const std::string& pct) {
      std::ostringstream oss;
      oss << "| " << std::left << std::setw(phaseW) << phase << " | " << std::right << std::setw(msW) << ms << " | "
          << std::left << std::setw(pctW) << pct << " |";
      lines.push_back(oss.str());
    };

    lines.push_back(" Timing - batch " + std::to_string(v.batchNumber) + " (current, ms/batch)");

    sep();

    row("phase", "ms/batch", "%");

    sep();

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = v.orch[p];
      const double pct = ms / total * 100.0;

      std::ostringstream msStr;
      msStr << std::fixed << std::setprecision(1) << ms;

      std::ostringstream pctStr;
      pctStr << std::fixed << std::setprecision(1) << pct << "%";

      row(phaseLabel(ph), msStr.str(), pctStr.str());

      if (ph == Phase::DataFetch) {
        const double augMs = v.orch[static_cast<int>(Phase::Augmentation)];

        if (augMs > 0.0) {
          std::ostringstream augStr;
          augStr << std::fixed << std::setprecision(1) << augMs;
          std::ostringstream augPctStr;
          augPctStr << std::fixed << std::setprecision(1) << augMs / total * 100.0 << "%";
          row(" + augmentation", augStr.str(), augPctStr.str());
        }
      }

      if (ph == Phase::GpuTrain) {
        std::ostringstream h2dStr;
        h2dStr << std::fixed << std::setprecision(1) << v.h2dPerGpu;
        std::ostringstream h2dPctStr;
        h2dPctStr << std::fixed << std::setprecision(1) << v.h2dPerGpu / total * 100.0 << "%";
        row("  h2d_upload", h2dStr.str(), h2dPctStr.str());

        std::ostringstream compStr;
        compStr << std::fixed << std::setprecision(1) << v.computePerGpu;
        std::ostringstream compPctStr;
        compPctStr << std::fixed << std::setprecision(1) << v.computePerGpu / total * 100.0 << "%";
        row("  gpu_compute", compStr.str(), compPctStr.str());

        const double gpuProfileTotal = gpuSubTotal(v.gpuProfile);

        if (gpuProfileTotal > 0.0) {
          auto gpuSub = [&](const char* label, double msVal) {
            if (msVal > 0.0) {
              std::ostringstream subStr;
              subStr << std::fixed << std::setprecision(1) << msVal;
              std::ostringstream subPctStr;
              subPctStr << std::fixed << std::setprecision(1) << msVal / total * 100.0 << "%";
              std::string indent = std::string("    ") + label;
              row(indent, subStr.str(), subPctStr.str());
            }
          };

          for (Phase ph : kGpuSubPhases)
            gpuSub(phaseLabel(ph), v.gpuProfile[static_cast<int>(ph)]);
        }
      }
    }

    sep();

    {
      std::ostringstream totalStr;
      totalStr << std::fixed << std::setprecision(1) << total;
      row("TOTAL", totalStr.str(), "");
    }

    sep();

    if (launchesPerGpu > 0) {
      std::ostringstream oss;
      oss << " gpu launches/batch per GPU: " << launchesPerGpu << " (" << std::fixed << std::setprecision(2)
          << msPerLaunch << " ms each)";
      lines.push_back(oss.str());
    }

    return lines;
  }

} // namespace NN_CLI
