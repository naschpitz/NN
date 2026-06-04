#include "NN-CLI_TrainingProfiler.hpp"

#include <algorithm>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <sstream>

#include <unistd.h>

namespace NN_CLI
{
  using Phase = CNN::TimingPhase;
  using Event = CNN::TimingEvent;

  namespace
  {
    volatile std::sig_atomic_t g_liveTableLines = 0;

    void writeRaw(const char* s, std::size_t n)
    {
      while (n > 0) {
        const ssize_t w = ::write(STDOUT_FILENO, s, n);

        if (w <= 0)
          break;

        s += w;
        n -= static_cast<std::size_t>(w);
      }
    }

    void onTerminateSignal(int sig)
    {
      const int k = g_liveTableLines;

      if (k > 0) {
        writeRaw("\r", 1);

        for (int i = 0; i < k; i++)
          writeRaw("\n\033[K", 4);

        for (int i = 0; i < k; i++)
          writeRaw("\033[A", 3);

        g_liveTableLines = 0;
      }

      std::signal(sig, SIG_DFL);
      std::raise(sig);
    }

    constexpr Phase kOrchPhases[] = {Phase::DataFetch, Phase::GpuTrain, Phase::GradMerge, Phase::WeightUpdate,
                                     Phase::KernelRestore};
  } // namespace

  //===================================================================================================================//

  TrainingProfiler::TrainingProfiler()
  {
    this->reset();
  }

  void TrainingProfiler::reset()
  {
    static bool handlersInstalled = false;

    if (!handlersInstalled) {
      std::signal(SIGINT, onTerminateSignal);
      std::signal(SIGTERM, onTerminateSignal);
      handlersInstalled = true;
    }

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
    this->lastRenderedLines = 0;
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
    this->numGpus = 1;
  }

  void TrainingProfiler::setNumGpus(int numGpus)
  {
    this->numGpus = std::max(1, numGpus);
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

    for (int g = 1; g <= this->numGpus && g < kMaxRows; g++) {
      h2dSum += this->stepMs[g][static_cast<int>(Phase::H2DUpload)];
      computeSum += this->stepMs[g][static_cast<int>(Phase::GpuCompute)];
      runs += this->stepRuns[g];
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

    {
      std::lock_guard<std::mutex> lock(this->mutex);
      this->lastStep = view;
    }
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

    std::lock_guard<std::mutex> lock(this->mutex);
    this->lastStep.valid = false;
  }

  bool TrainingProfiler::hasData() const
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    return this->lastStep.valid;
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
    default:
      return "?";
    }
  }

  std::string TrainingProfiler::fmt(double v, int width, int prec)
  {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << std::setw(width) << v;
    return o.str();
  }

  //===================================================================================================================//
  //-- Legacy std::ostream rendering --//
  //===================================================================================================================//

  void TrainingProfiler::renderLiveTable(std::ostream& out)
  {
    StepView v;
    {
      std::lock_guard<std::mutex> lock(this->mutex);

      if (!this->lastStep.valid)
        return;

      if (this->lastStep.batchNumber == this->lastRenderedBatchNumber)
        return;

      v = this->lastStep;
    }

    const double total = v.orchTotal > 0.0 ? v.orchTotal : 1.0;
    const ulong launchesPerGpu = v.runs / static_cast<ulong>(std::max(1, this->numGpus));
    const double msPerLaunch = launchesPerGpu > 0 ? v.computePerGpu / static_cast<double>(launchesPerGpu) : 0.0;

    std::ostringstream t;
    int lineCount = 0;
    auto line = [&](const std::string& s) {
      t << "\n\033[K" << s;
      lineCount++;
    };

    line("");
    line("  Timing - batch " + std::to_string(v.batchNumber) + " (current, ms/batch)");
    line("  +----------------+-----------+--------+");
    line("  | phase          |  ms/batch |      % |");
    line("  +----------------+-----------+--------+");

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = v.orch[p];
      const double pct = ms / total * 100.0;
      std::ostringstream row;
      row << "  | " << std::left << std::setw(14) << phaseLabel(ph) << std::right << " | " << fmt(ms, 9) << " | "
          << fmt(pct, 5) << " %|";
      line(row.str());

      if (ph == Phase::GpuTrain) {
        std::ostringstream h2d;
        h2d << "  |   + h2d_upload | " << fmt(v.h2dPerGpu, 9) << " |   (gpu)|";
        line(h2d.str());

        std::ostringstream comp;
        comp << "  |   + gpu_compute| " << fmt(v.computePerGpu, 9) << " |   (gpu)|";
        line(comp.str());
      }
    }

    line("  +----------------+-----------+--------+");
    std::ostringstream tot;
    tot << "  | " << std::left << std::setw(14) << "TOTAL" << std::right << " | " << fmt(total, 9) << " |        |";
    line(tot.str());
    line("  +----------------+-----------+--------+");

    if (launchesPerGpu > 0)
      line("  gpu launches/batch per GPU: " + fmt(launchesPerGpu, 0, 0) + "  (one run() per sample, " +
           fmt(msPerLaunch, 0, 2) + " ms each)");

    t << "\033[" << lineCount << "A";

    this->lastRenderedLines = lineCount;
    this->lastRenderedBatchNumber = v.batchNumber;
    g_liveTableLines = lineCount;

    out << t.str();
    out.flush();
  }

  void TrainingProfiler::clearLiveTable(std::ostream& out)
  {
    if (this->lastRenderedLines <= 0)
      return;

    std::ostringstream t;

    for (int i = 0; i < this->lastRenderedLines; i++)
      t << "\n\033[K";

    t << "\033[" << this->lastRenderedLines << "A";

    out << t.str();
    out.flush();
    this->lastRenderedLines = 0;
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
    g_liveTableLines = 0;
  }

  void TrainingProfiler::renderEpochSummary(std::ostream& out, ulong epoch)
  {
    std::array<double, kNumPhases> ep;
    ulong steps;
    ulong runs;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      ep = this->epochMs;
      steps = this->stepCount;
      runs = this->epochRuns;
    }

    if (steps == 0)
      return;

    double orchTotal = 0.0;

    for (Phase ph : kOrchPhases)
      orchTotal += ep[static_cast<int>(ph)];

    if (orchTotal <= 0.0)
      return;

    const double n = static_cast<double>(steps);
    const int gpus = std::max(1, this->numGpus);

    out << "\n  Timing summary - epoch " << epoch << " (" << steps << " batches, avg ms/batch)\n";
    out << "  +----------------+-----------+--------+\n";
    out << "  | phase          |  ms/batch |      % |\n";
    out << "  +----------------+-----------+--------+\n";

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = ep[p] / n;
      const double pct = ep[p] / orchTotal * 100.0;
      out << "  | " << std::left << std::setw(14) << phaseLabel(ph) << std::right << " | " << fmt(ms, 9) << " | "
          << fmt(pct, 5) << " %|\n";

      if (ph == Phase::GpuTrain) {
        const double h2d = ep[static_cast<int>(Phase::H2DUpload)] / n / gpus;
        const double comp = ep[static_cast<int>(Phase::GpuCompute)] / n / gpus;
        out << "  |   + h2d_upload | " << fmt(h2d, 9) << " |   (gpu)|\n";
        out << "  |   + gpu_compute| " << fmt(comp, 9) << " |   (gpu)|\n";
      }
    }

    out << "  +----------------+-----------+--------+\n";
    out << "  | " << std::left << std::setw(14) << "TOTAL" << std::right << " | " << fmt(orchTotal / n, 9)
        << " |        |\n";
    out << "  +----------------+-----------+--------+\n";

    if (runs > 0) {
      const double launchesPerStep = static_cast<double>(runs) / n / gpus;
      out << "  gpu launches/batch per GPU: " << fmt(launchesPerStep, 0, 0)
          << "  (one run() per sample in the fast path)\n";
    }
  }

  void TrainingProfiler::renderFinalSummary(std::ostream& out)
  {
    if (this->stepInProgress) {
      this->finalizeStep();
      this->stepInProgress = false;
    }

    std::array<double, kNumPhases> tot;
    ulong steps;
    ulong runs;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      tot = this->totalMs;
      steps = this->totalStepCount;
      runs = this->totalRuns;
    }

    if (steps == 0)
      return;

    double orchTotal = 0.0;

    for (Phase ph : kOrchPhases)
      orchTotal += tot[static_cast<int>(ph)];

    if (orchTotal <= 0.0)
      return;

    const int gpus = std::max(1, this->numGpus);

    out << "\n  Timing summary - whole run (" << steps << " batches total)\n";
    out << "  +----------------+---------------+--------+\n";
    out << "  | phase          |     total (s) |      % |\n";
    out << "  +----------------+---------------+--------+\n";

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double s = tot[p] / 1000.0;
      const double pct = tot[p] / orchTotal * 100.0;
      out << "  | " << std::left << std::setw(14) << phaseLabel(ph) << std::right << " | " << fmt(s, 13, 2) << " | "
          << fmt(pct, 5) << " %|\n";

      if (ph == Phase::GpuTrain) {
        const double h2d = tot[static_cast<int>(Phase::H2DUpload)] / 1000.0 / gpus;
        const double comp = tot[static_cast<int>(Phase::GpuCompute)] / 1000.0 / gpus;
        out << "  |   + h2d_upload | " << fmt(h2d, 13, 2) << " |   (gpu)|\n";
        out << "  |   + gpu_compute| " << fmt(comp, 13, 2) << " |   (gpu)|\n";
      }
    }

    out << "  +----------------+---------------+--------+\n";
    out << "  | " << std::left << std::setw(14) << "TOTAL" << std::right << " | " << fmt(orchTotal / 1000.0, 13, 2)
        << " |        |\n";
    out << "  +----------------+---------------+--------+\n";

    if (runs > 0) {
      const double launchesPerStep = static_cast<double>(runs) / static_cast<double>(steps) / gpus;
      out << "  gpu launches/batch per GPU: " << fmt(launchesPerStep, 0, 0)
          << " - each launch is a single-sample fused forward+backward run()\n";
    }
  }

  //===================================================================================================================//
  //-- TUI table lines (for CDK label) --//
  //===================================================================================================================//

  std::vector<std::string> TrainingProfiler::getTimingLines() const
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

    // Fixed column widths matching the live table format
    constexpr int phaseW = 16;
    constexpr int msW = 11;
    constexpr int pctW = 8;

    auto sep = [&]() {
      lines.push_back("+" + std::string(phaseW + 1, '-') + "+" + std::string(msW + 1, '-') + "+" +
                      std::string(pctW + 1, '-') + "+");
    };

    char buf[128];
    snprintf(buf, sizeof(buf), " Timing - batch %lu (current, ms/batch)", static_cast<unsigned long>(v.batchNumber));
    lines.push_back(buf);

    sep();

    snprintf(buf, sizeof(buf), "| %-14s  | %*s | %*s |", "phase", msW - 1, "ms/batch", pctW - 1, "%");
    lines.push_back(buf);

    sep();

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = v.orch[p];
      const double pct = ms / total * 100.0;

      snprintf(buf, sizeof(buf), "| %-14s  | %*.1f | %5.1f%% |", phaseLabel(ph), msW - 1, ms, pct);
      lines.push_back(buf);

      if (ph == Phase::GpuTrain) {
        snprintf(buf, sizeof(buf), "|   + h2d_upload | %*.1f |  (gpu) |", msW - 1, v.h2dPerGpu);
        lines.push_back(buf);
        snprintf(buf, sizeof(buf), "|   + gpu_compute| %*.1f |  (gpu) |", msW - 1, v.computePerGpu);
        lines.push_back(buf);
      }
    }

    sep();

    snprintf(buf, sizeof(buf), "| %-14s  | %*.1f | %*s |", "TOTAL", msW - 1, total, pctW - 1, "");
    lines.push_back(buf);

    sep();

    if (launchesPerGpu > 0) {
      snprintf(buf, sizeof(buf), " gpu launches/batch per GPU: %lu (%.2f ms each)", launchesPerGpu, msPerLaunch);
      lines.push_back(buf);
    }

    return lines;
  }

} // namespace NN_CLI
