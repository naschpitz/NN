#include "NN-CLI_TrainingProfiler.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace NN_CLI
{
  using Phase = CNN::TimingPhase;
  using Event = CNN::TimingEvent;

  namespace
  {
    // Orchestrator phases, in display order. These are sequential / non-overlapping
    // and together account for the whole batch wall time (the % denominator).
    constexpr Phase kOrchPhases[] = {Phase::DataFetch, Phase::GpuTrain, Phase::GradMerge, Phase::WeightUpdate,
                                     Phase::KernelRestore};

    std::string fmt(double v, int width, int prec = 1)
    {
      std::ostringstream o;
      o << std::fixed << std::setprecision(prec) << std::setw(width) << v;
      return o.str();
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
    this->lastRenderedLines = 0;
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
    this->numGpus = 1; // re-detected from events
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

    // Auto-detect the GPU count from the events themselves (config may say "0 = all").
    if (gpuIndex >= 0 && gpuIndex + 1 > this->numGpus)
      this->numGpus = gpuIndex + 1;

    if (event == Event::Begin) {
      // A new mini-batch starts at DataFetch-Begin on the orchestrator row.
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
  //-- Step finalize: roll up + publish snapshot --//
  //===================================================================================================================//

  void TrainingProfiler::finalizeStep()
  {
    // Sum worker phases across active GPU rows (1..numGpus).
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

    // Roll into epoch + total accumulators.
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

    // Reset per-step state.
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

    // The previous epoch's step accumulators are already rolled into epochMs as each
    // step finalized. Reset per-epoch accumulators for the new epoch.
    this->currentEpoch = epoch;
    this->epochMs.fill(0.0);
    this->epochRuns = 0;
    this->stepCount = 0;
  }

  bool TrainingProfiler::hasData() const
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    return this->lastStep.valid;
  }

  //===================================================================================================================//
  //-- Rendering --//
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

  void TrainingProfiler::renderLiveTable(std::ostream& out)
  {
    StepView v;
    {
      std::lock_guard<std::mutex> lock(this->mutex);

      if (!this->lastStep.valid)
        return;

      // The table only changes when a new step is finalized. Skip redraws that would
      // just paint the same numbers — the training callback fires per-sample (up to
      // 128 k times per epoch) while a step lasts tens of seconds.
      if (this->lastStep.batchNumber == this->lastRenderedBatchNumber)
        return;

      v = this->lastStep;
    }

    const double total = v.orchTotal > 0.0 ? v.orchTotal : 1.0;
    const ulong launchesPerGpu = v.runs / static_cast<ulong>(std::max(1, this->numGpus));
    const double msPerLaunch = launchesPerGpu > 0 ? v.computePerGpu / static_cast<double>(launchesPerGpu) : 0.0;

    // Write the timing table lines below the progress bar (the bar was just drawn
    // by progressBar.update() on the current line). Use explicit relative cursor
    // movement — no \033[s / \033[u — so the table never corrupts the bar line or
    // leaks into an unrelated screen region.
    std::ostringstream t;

    int lineCount = 0;
    auto line = [&](const std::string& s) {
      t << "\n\033[K" << s;
      lineCount++;
    };

    line("  ┌─ timings (batch " + std::to_string(v.batchNumber) + ", ms/batch) " + std::string(28, '-'));

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = v.orch[p];
      const double pct = ms / total * 100.0;
      std::ostringstream row;
      row << "  │ " << std::left << std::setw(16) << phaseLabel(ph) << std::right << fmt(ms, 9) << " ms  "
          << fmt(pct, 5) << " %";
      line(row.str());

      if (ph == Phase::GpuTrain) {
        std::ostringstream h2d;
        h2d << "  │   ├ " << std::left << std::setw(14) << "h2d_upload" << std::right << fmt(v.h2dPerGpu, 9)
            << " ms   (per-GPU, async)";
        line(h2d.str());

        std::ostringstream comp;
        comp << "  │   └ " << std::left << std::setw(14) << "gpu_compute" << std::right << fmt(v.computePerGpu, 9)
             << " ms   (per-GPU, fwd+bwd; " << launchesPerGpu << " launches @ " << fmt(msPerLaunch, 0, 2) << " ms)";
        line(comp.str());
      }
    }

    std::ostringstream tot;
    tot << "  │ " << std::left << std::setw(16) << "TOTAL" << std::right << fmt(total, 9) << " ms";
    line(tot.str());
    line("  └" + std::string(46, '-'));

    // Move cursor back up to the progress-bar line so the next \r from
    // progressBar.update() lands on the bar, not inside the timing block.
    t << "\033[" << lineCount << "A";

    this->lastRenderedLines = lineCount;
    this->lastRenderedBatchNumber = v.batchNumber;

    out << t.str();
    out.flush();
  }

  void TrainingProfiler::clearLiveTable(std::ostream& out)
  {
    if (this->lastRenderedLines <= 0)
      return;

    std::ostringstream t;

    for (int i = 0; i < this->lastRenderedLines; i++)
      t << "\r\033[K\n";

    t << "\033[" << this->lastRenderedLines << "A";

    out << t.str();
    out.flush();
    this->lastRenderedLines = 0;
    this->lastRenderedBatchNumber = static_cast<ulong>(-1);
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

    out << "\n  Timing summary — epoch " << epoch << " (" << steps << " batches, avg ms/batch)\n";
    out << "  ┌────────────────┬───────────┬────────┐\n";
    out << "  │ phase          │  ms/batch │      % │\n";
    out << "  ├────────────────┼───────────┼────────┤\n";

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double ms = ep[p] / n;
      const double pct = ep[p] / orchTotal * 100.0;
      out << "  │ " << std::left << std::setw(14) << phaseLabel(ph) << std::right << " │ " << fmt(ms, 9) << " │ "
          << fmt(pct, 5) << " %│\n";

      if (ph == Phase::GpuTrain) {
        const double h2d = ep[static_cast<int>(Phase::H2DUpload)] / n / gpus;
        const double comp = ep[static_cast<int>(Phase::GpuCompute)] / n / gpus;
        out << "  │   ├ h2d_upload  │ " << fmt(h2d, 9) << " │   (gpu)│\n";
        out << "  │   └ gpu_compute│ " << fmt(comp, 9) << " │   (gpu)│\n";
      }
    }

    out << "  ├────────────────┼───────────┼────────┤\n";
    out << "  │ " << std::left << std::setw(14) << "TOTAL" << std::right << " │ " << fmt(orchTotal / n, 9)
        << " │        │\n";
    out << "  └────────────────┴───────────┴────────┘\n";

    if (runs > 0) {
      const double launchesPerStep = static_cast<double>(runs) / n / gpus;
      out << "  gpu launches/batch per GPU: " << fmt(launchesPerStep, 0, 0)
          << "  (one run() per sample in the fast path)\n";
    }
  }

  void TrainingProfiler::renderFinalSummary(std::ostream& out)
  {
    // Finalize any dangling step (last batch never sees a following DataFetch-Begin).
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

    out << "\n  Timing summary — whole run (" << steps << " batches total)\n";
    out << "  ┌────────────────┬───────────────┬────────┐\n";
    out << "  │ phase          │     total (s) │      % │\n";
    out << "  ├────────────────┼───────────────┼────────┤\n";

    for (Phase ph : kOrchPhases) {
      const int p = static_cast<int>(ph);
      const double s = tot[p] / 1000.0;
      const double pct = tot[p] / orchTotal * 100.0;
      out << "  │ " << std::left << std::setw(14) << phaseLabel(ph) << std::right << " │ " << fmt(s, 13, 2) << " │ "
          << fmt(pct, 5) << " %│\n";

      if (ph == Phase::GpuTrain) {
        const double h2d = tot[static_cast<int>(Phase::H2DUpload)] / 1000.0 / gpus;
        const double comp = tot[static_cast<int>(Phase::GpuCompute)] / 1000.0 / gpus;
        out << "  │   ├ h2d_upload  │ " << fmt(h2d, 13, 2) << " │   (gpu)│\n";
        out << "  │   └ gpu_compute│ " << fmt(comp, 13, 2) << " │   (gpu)│\n";
      }
    }

    out << "  ├────────────────┼───────────────┼────────┤\n";
    out << "  │ " << std::left << std::setw(14) << "TOTAL" << std::right << " │ " << fmt(orchTotal / 1000.0, 13, 2)
        << " │        │\n";
    out << "  └────────────────┴───────────────┴────────┘\n";

    if (runs > 0) {
      const double launchesPerStep = static_cast<double>(runs) / static_cast<double>(steps) / gpus;
      out << "  gpu launches/batch per GPU: " << fmt(launchesPerStep, 0, 0)
          << " — each launch is a single-sample fused forward+backward run()\n";
    }
  }

} // namespace NN_CLI
