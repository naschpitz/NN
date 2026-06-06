#include "NN-CLI_ProgressBar.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <curses.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
  //-- Layout constants shared by every ncurses progress / loading / validation bar so they all
  //-- line up vertically in the Training panel. --//
  constexpr int kLabelWidth = 28; // labels are left-padded to this width
  constexpr int kBracketWidth = 2; // the " [" written right after the label
  constexpr int kRightPad = 20; // reserved on the right for the "] 100.000%" suffix (+ slack)
  constexpr int kGpuInfoPerGpu = 9; // width of one " | N:XXX%" cell in the per-GPU suffix

  //-- Color pairs (configured in TerminalUI::init). --//
  constexpr int kBarColor = 1; // green filled bar segment
  constexpr int kLossColor = 6; // green epoch-complete loss text

  std::string formatEta(double seconds)
  {
    if (seconds < 0.0)
      seconds = 0.0;

    long total = static_cast<long>(seconds + 0.5);
    long h = total / 3600;
    long m = (total % 3600) / 60;
    long s = total % 60;

    std::ostringstream out;

    if (h > 0)
      out << h << ":" << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;
    else
      out << std::setfill('0') << std::setw(2) << m << ":" << std::setw(2) << s;

    return out.str();
  }

  // Width reserved on the right for the per-GPU "(0:XX% | 1:XX%)" suffix (0 when single-GPU).
  int gpuSuffixWidth(int gpus)
  {
    return gpus > 1 ? kGpuInfoPerGpu * gpus - 1 : 0;
  }

  // Usable bar width inside `win`, mirrored across all bars so their brackets line up.
  int barWidthFor(WINDOW* win, int gpus)
  {
    int cols = ::getmaxx(win);
    return std::max(10, cols - kLabelWidth - kBracketWidth - kRightPad - gpuSuffixWidth(gpus));
  }

  // Emit the label left-padded to kLabelWidth, then " [".
  void emitLabel(WINDOW* win, const char* label)
  {
    int len = static_cast<int>(std::strlen(label));
    ::waddstr(win, label);

    for (int i = len; i < kLabelWidth; i++)
      ::waddstr(win, " ");

    ::waddstr(win, " [");
  }

  // Emit a single solid bar: `filled` green blocks, then light shading out to `barWidth`.
  void emitSingleBar(WINDOW* win, float fraction, int barWidth)
  {
    int filled = std::clamp(static_cast<int>(fraction * barWidth), 0, barWidth);

    ::wattron(win, COLOR_PAIR(kBarColor));

    for (int i = 0; i < filled; i++)
      ::waddstr(win, "█");

    ::wattroff(win, COLOR_PAIR(kBarColor));

    for (int i = filled; i < barWidth; i++)
      ::waddstr(win, "░");
  }

  // Emit a segmented bar, one equal-width segment per GPU separated by "│".
  void emitMultiBar(WINDOW* win, const std::vector<float>& gpuProg, int numGPUs, int barWidth)
  {
    int segmentWidth = barWidth / numGPUs;

    for (int gpu = 0; gpu < numGPUs; gpu++) {
      float pct = (gpu < static_cast<int>(gpuProg.size())) ? gpuProg[gpu] : 0.0f;
      emitSingleBar(win, pct, segmentWidth);

      if (gpu < numGPUs - 1)
        ::waddstr(win, "│");
    }
  }

  // Emit the trailing per-GPU suffix " (0:XX% | 1:XX%)".
  void emitGpuSuffix(WINDOW* win, const std::vector<float>& gpuProg)
  {
    std::ostringstream gpuInfo;
    gpuInfo << " (";

    for (size_t g = 0; g < gpuProg.size(); g++) {
      gpuInfo << g << ":" << std::setw(3) << static_cast<int>(gpuProg[g] * 100) << "%";

      if (g + 1 < gpuProg.size())
        gpuInfo << " | ";
    }

    gpuInfo << ")";
    ::waddstr(win, gpuInfo.str().c_str());
  }
} // namespace

namespace NN_CLI
{

  ProgressBar::ProgressBar(ulong progressReports, int barWidth, ulong windowSize)
    : progressReports(progressReports),
      barWidth(barWidth),
      windowSize(windowSize)
  {
  }

  //===================================================================================================================//

  void ProgressBar::update(const ProgressInfo& progress, WINDOW* win)
  {
    bool isEpochComplete = (progress.epochLoss > 0);
    bool isMultiGPU = (progress.totalGPUs > 1);

    // Reset per-epoch timer / loss when a new epoch begins.
    if (!isEpochComplete && this->timerEpoch != progress.currentEpoch) {
      this->timerEpoch = progress.currentEpoch;
      this->epochStartTime = std::chrono::steady_clock::now();
      this->runningLossSum = 0.0;
      this->runningLossCount = 0;
      this->rateWindow.clear();
    }

    // Reset GPU state at the start of each epoch and render 0% bar immediately
    if (progress.gpuIndex >= 0 && this->currentEpoch != progress.currentEpoch) {
      this->resetGpuState(progress.totalGPUs, progress.currentEpoch);

      if (!win) {
        std::ostringstream out;
        out << "\r\033[KEpoch " << std::setw(4) << progress.currentEpoch << "/" << progress.totalEpochs << " [";

        if (isMultiGPU) {
          std::vector<float> zeroProg(progress.totalGPUs, 0.0f);
          this->renderMultiGpuBar(out, zeroProg, progress.totalGPUs);
        } else {
          this->renderSingleBar(out, 0.0f);
        }

        out << " - Loss: " << std::fixed << std::setprecision(6) << 0.0f << "   ";
        std::cout << out.str() << std::flush;
      } else {
        // ncurses: render the 0% bar into the window immediately at epoch start (before throttling).
        ::werase(win);

        char rawLabel[32];
        snprintf(rawLabel, sizeof(rawLabel), "Epoch %*lu/%lu", 4, static_cast<unsigned long>(progress.currentEpoch),
                 static_cast<unsigned long>(progress.totalEpochs));
        emitLabel(win, rawLabel);

        int cols0 = ::getmaxx(win);
        int overhead = kLabelWidth + kBracketWidth + kRightPad + gpuSuffixWidth(progress.totalGPUs);
        int bw = std::max(10, cols0 - overhead);

        if (isMultiGPU) {
          std::vector<float> zeroProg(progress.totalGPUs, 0.0f);
          emitMultiBar(win, zeroProg, progress.totalGPUs, bw);
          ::waddstr(win, "]   0.000%");

          if (gpuSuffixWidth(progress.totalGPUs) > 0 && bw + overhead <= cols0)
            emitGpuSuffix(win, zeroProg);
        } else {
          emitSingleBar(win, 0.0f, bw);
          ::waddstr(win, "]   0.000%");
        }

        ::wmove(win, 1, 0);
        ::waddstr(win, "Loss: 0.000000                                        ");
        ::wnoutrefresh(win);
      }
    }

    // For multi-GPU, update per-GPU progress
    if (isMultiGPU && progress.gpuIndex >= 0) {
      ulong samplesPerGPU = progress.totalSamples / progress.totalGPUs;
      float gpuPercent = static_cast<float>(progress.currentSample) / samplesPerGPU;
      gpuPercent = std::min(1.0f, std::max(0.0f, gpuPercent));
      this->updateGpuProgress(progress.gpuIndex, gpuPercent);
    }

    // Accumulate running loss
    if (!isEpochComplete) {
      this->runningLossSum += static_cast<double>(progress.sampleLoss);
      this->runningLossCount++;
    }

    // Throttle output
    if (!isEpochComplete && this->progressReports > 0) {
      ulong interval = std::max(static_cast<ulong>(1), progress.totalSamples / this->progressReports);

      if (progress.currentSample % interval != 0 && progress.currentSample != progress.totalSamples)
        return;
    } else if (!isEpochComplete && this->progressReports == 0) {
      return;
    }

    // Compute common values
    double runningAvg = this->runningLossCount > 0 ? this->runningLossSum / this->runningLossCount : 0.0;

    double fractionDone;

    if (isMultiGPU) {
      std::vector<float> gpuProg = this->getGpuProgress();
      double sum = 0.0;

      for (float p : gpuProg)
        sum += p;

      fractionDone = gpuProg.empty() ? 0.0 : sum / static_cast<double>(gpuProg.size());
    } else {
      fractionDone =
        (progress.totalSamples > 0) ? static_cast<double>(progress.currentSample) / progress.totalSamples : 0.0;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - this->epochStartTime).count();
    double samplesDone = fractionDone * static_cast<double>(progress.totalSamples);

    // Record a sample for sliding-window rate calculation (skip epoch-complete frames)
    if (this->windowSize > 0 && !isEpochComplete) {
      this->rateWindow.push_back({samplesDone, now});

      if (this->rateWindow.size() > this->windowSize)
        this->rateWindow.pop_front();
    }

    // Compute rate using sliding window (or fallback to full-epoch)
    double rate = 0.0;

    if (this->windowSize > 0 && this->rateWindow.size() >= 2) {
      const auto& oldest = this->rateWindow.front();
      const auto& newest = this->rateWindow.back();
      double sampleDelta = newest.samplesProcessed - oldest.samplesProcessed;
      double timeDelta = std::chrono::duration<double>(newest.timestamp - oldest.timestamp).count();
      rate = (timeDelta > 0.0) ? sampleDelta / timeDelta : 0.0;
    } else {
      rate = (elapsed > 0.0) ? samplesDone / elapsed : 0.0;
    }

    double eta = (rate > 0.0) ? (static_cast<double>(progress.totalSamples) - samplesDone) / rate : 0.0;

    if (win) {
      //--- ncurses rendering ---//
      ::werase(win);

      int cols = ::getmaxx(win);

      // Reserve the per-GPU suffix width whenever multi-GPU — including the epoch-complete
      // frame, even though it shows no suffix. Otherwise the completed bar would render
      // full-width and its "]" would jut ~gpuSuffixWidth columns right of the loading bar's "]",
      // which keeps reserving this space. The reservation keeps both brackets aligned; the
      // suffix itself is still only rendered while the epoch is in progress (see below).
      int overhead = kLabelWidth + kBracketWidth + kRightPad + gpuSuffixWidth(progress.totalGPUs);
      int effBarWidth = std::max(10, cols - overhead);

      // Snapshot per-GPU progress once so the bar, percentage and suffix stay consistent.
      std::vector<float> gpuProg;

      if (isMultiGPU)
        gpuProg = this->getGpuProgress();

      // Line 0: epoch label + progress bar [+ per-GPU percentages inline]
      char rawLabel[32];
      snprintf(rawLabel, sizeof(rawLabel), "Epoch %*lu/%lu", 4, static_cast<unsigned long>(progress.currentEpoch),
               static_cast<unsigned long>(progress.totalEpochs));
      emitLabel(win, rawLabel);

      if (isMultiGPU && !isEpochComplete) {
        emitMultiBar(win, gpuProg, progress.totalGPUs, effBarWidth);
      } else {
        float samplePercent =
          isEpochComplete ? 1.0f : static_cast<float>(progress.currentSample) / progress.totalSamples;
        emitSingleBar(win, samplePercent, effBarWidth);
      }

      float displayPct;

      if (isMultiGPU) {
        double sum = 0.0;

        for (float p : gpuProg)
          sum += p;

        displayPct = gpuProg.empty() ? 0.0f : static_cast<float>(sum / gpuProg.size()) * 100.0f;
      } else {
        displayPct =
          isEpochComplete ? 100.0f : static_cast<float>(progress.currentSample) / progress.totalSamples * 100.0f;
      }

      char pctBuf[16];
      snprintf(pctBuf, sizeof(pctBuf), "] %6.3f%%", static_cast<double>(displayPct));
      ::waddstr(win, pctBuf);

      if (isMultiGPU && !isEpochComplete && gpuSuffixWidth(progress.totalGPUs) > 0 && effBarWidth + overhead <= cols)
        emitGpuSuffix(win, gpuProg);

      // Line 1: loss, img/s, ETA
      if (isEpochComplete) {
        ::wmove(win, 1, 0);
        ::wattron(win, COLOR_PAIR(kLossColor));
        char lossBuf[64];
        snprintf(lossBuf, sizeof(lossBuf), "Loss: %.6f ", static_cast<double>(progress.epochLoss));
        ::waddstr(win, lossBuf);
        ::wattroff(win, COLOR_PAIR(kLossColor));
      } else {
        ::wmove(win, 1, 0);

        char statsBuf[128];
        snprintf(statsBuf, sizeof(statsBuf), "Loss: %.6f  %6ld img/s  ETA %s   ", static_cast<double>(runningAvg),
                 static_cast<long>(rate), formatEta(eta).c_str());
        ::waddstr(win, statsBuf);
      }

      ::wnoutrefresh(win);
    } else {
      //--- std::cout rendering (legacy path) ---//
      std::ostringstream out;
      out << "\r\033[KEpoch " << std::setw(4) << progress.currentEpoch << "/" << progress.totalEpochs << " [";

      if (isMultiGPU && !isEpochComplete) {
        std::vector<float> gpuProg = this->getGpuProgress();
        this->renderMultiGpuBar(out, gpuProg, progress.totalGPUs);
      } else {
        float samplePercent =
          isEpochComplete ? 1.0f : static_cast<float>(progress.currentSample) / progress.totalSamples;
        this->renderSingleBar(out, samplePercent);
      }

      if (isEpochComplete) {
        out << " - Loss: " << std::fixed << std::setprecision(6) << progress.epochLoss;

        if (!this->holdEpochLine)
          out << std::endl;
      } else {
        out << " - Loss: " << std::fixed << std::setprecision(6) << runningAvg;
        out << " - " << std::setw(6) << static_cast<long>(rate) << " img/s - ETA " << formatEta(eta) << "   ";
      }

      std::cout << out.str() << std::flush;
    }
  }

  //===================================================================================================================//

  void ProgressBar::printLoadingProgress(const std::string& label, size_t current, size_t total, ulong progressReports,
                                         int barWidth)
  {
    ulong interval = (progressReports > 0) ? std::max(static_cast<size_t>(1), total / progressReports) : 0;

    if (interval == 0)
      return;

    if (current > 1 && current != total && (current % interval) != 0)
      return;

    float percent = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    int filledWidth = static_cast<int>(percent * barWidth);

    std::ostringstream out;
    out << "\r" << label << " [";

    for (int i = 0; i < barWidth; i++)
      out << (i < filledWidth ? "█" : "░");

    out << "] " << current << "/" << total << "  " << std::fixed << std::setprecision(1) << (percent * 100.0f) << "%";
    out << "   ";

    std::cout << out.str() << std::flush;

    if (current == total)
      std::cout << std::endl;
  }

  void ProgressBar::renderValidationBar(WINDOW* win, float pct, int numGpus)
  {
    if (!win)
      return;

    int barWidth = barWidthFor(win, numGpus);

    ::werase(win);
    emitLabel(win, "Validating");
    emitSingleBar(win, pct / 100.0f, barWidth);

    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), "] %6.3f%%", static_cast<double>(pct));
    ::waddstr(win, pctBuf);

    // Clear line 1 (epoch stats) during validation
    ::wmove(win, 1, 0);
    ::wclrtoeol(win);

    ::wnoutrefresh(win);
    ::doupdate();
  }

  void ProgressBar::renderLoadingBar(WINDOW* win, ulong current, ulong total, ulong batchNum, ulong totalBatches,
                                     int numGpus)
  {
    if (!win)
      return;

    float pct = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;

    char rawLabel[32];
    snprintf(rawLabel, sizeof(rawLabel), "Loading samples (%lu/%lu)", static_cast<unsigned long>(batchNum),
             static_cast<unsigned long>(totalBatches));

    // Mirror the epoch bar's geometry so the two bars line up vertically. numGpus is supplied by
    // the caller (the prefetch loader runs ahead of the first training update, so it cannot be
    // discovered dynamically here).
    int barWidth = barWidthFor(win, numGpus);

    ::werase(win);
    emitLabel(win, rawLabel);
    emitSingleBar(win, pct, barWidth);

    char buf[64];
    snprintf(buf, sizeof(buf), "] %lu/%lu  %5.1f%%", static_cast<unsigned long>(current),
             static_cast<unsigned long>(total), static_cast<double>(pct * 100.0));
    ::waddstr(win, buf);

    ::wnoutrefresh(win);
    ::doupdate();
  }

  //===================================================================================================================//
  //-- GPU State --//
  //===================================================================================================================//

  void ProgressBar::resetGpuState(int numGPUs, ulong epoch)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->totalGPUs = numGPUs;
    this->currentEpoch = epoch;
    this->gpuProgress.assign(numGPUs, 0.0f);
  }

  void ProgressBar::updateGpuProgress(int gpuIndex, float percent)
  {
    std::lock_guard<std::mutex> lock(this->mutex);

    if (gpuIndex >= 0 && gpuIndex < static_cast<int>(this->gpuProgress.size()))
      this->gpuProgress[gpuIndex] = percent;
  }

  std::vector<float> ProgressBar::getGpuProgress()
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    return this->gpuProgress;
  }

  //===================================================================================================================//
  //-- std::cout Rendering --//
  //===================================================================================================================//

  void ProgressBar::renderSingleBar(std::ostream& out, float percent)
  {
    int filledWidth = static_cast<int>(percent * this->barWidth);

    for (int i = 0; i < this->barWidth; i++)
      out << (i < filledWidth ? "█" : "░");

    out << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (percent * 100) << "%";
  }

  void ProgressBar::renderMultiGpuBar(std::ostream& out, const std::vector<float>& gpuProg, int numGPUs)
  {
    int segmentWidth = this->barWidth / numGPUs;

    for (int gpu = 0; gpu < numGPUs; gpu++) {
      float gpuPercent = (gpu < static_cast<int>(gpuProg.size())) ? gpuProg[gpu] : 0.0f;
      int filledWidth = static_cast<int>(gpuPercent * segmentWidth);

      for (int i = 0; i < segmentWidth; i++)
        out << (i < filledWidth ? "█" : "░");

      if (gpu < numGPUs - 1)
        out << "│";
    }

    float totalPercent = 0.0f;

    for (float p : gpuProg)
      totalPercent += p;

    if (numGPUs > 0)
      totalPercent /= numGPUs;

    out << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (totalPercent * 100) << "% (";
  }

} // namespace NN_CLI
