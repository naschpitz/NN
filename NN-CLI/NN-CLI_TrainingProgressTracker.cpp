#include "NN-CLI_TrainingProgressTracker.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace
{
  //-- Color pairs (configured in TerminalUI::init). --//
  constexpr int kLossColor = 6; // green epoch-complete loss text

  //===================================================================================================================//

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

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TrainingProgressTracker::TrainingProgressTracker(ulong progressReports, ulong windowSize, int barWidth)
    : progressReports(progressReports),
      barWidth(barWidth),
      windowSize(windowSize)
  {
  }

  //===================================================================================================================//
  //-- Public API --//
  //===================================================================================================================//

  void TrainingProgressTracker::update(const ProgressInfo& progress, WINDOW* win)
  {
    bool isEpochComplete = (progress.epochLoss > 0);
    bool isMultiSegment = (progress.totalGPUs > 1);

    //-- Reset per-epoch timer / loss / rate window when a new epoch begins. --//
    if (!isEpochComplete && this->timerEpoch != progress.currentEpoch) {
      this->timerEpoch = progress.currentEpoch;
      this->epochStartTime = std::chrono::steady_clock::now();
      this->runningLossSum = 0.0;
      this->runningLossCount = 0;
      this->rateWindow.clear();
    }

    //-- Reset segment state at the start of each epoch and render 0% bar immediately. --//
    if (progress.gpuIndex >= 0 && this->currentEpoch != progress.currentEpoch) {
      this->resetSegmentState(progress.totalGPUs, progress.currentEpoch);

      std::ostringstream labelStream;
      labelStream << "Epoch " << std::setw(4) << progress.currentEpoch << "/" << progress.totalEpochs;
      std::string label = labelStream.str();

      //-- ncurses epoch-start 0% bar --//
      if (isMultiSegment) {
        std::vector<float> zeroProg(progress.totalGPUs, 0.0f);
        this->renderer.renderMultiBar(win, label, zeroProg);
      } else {
        this->renderer.renderSingleBar(win, label, 0.0f);
      }

      this->renderer.renderSubLine(win, "Loss: 0.000000");
    }

    //-- Multi-segment progress update. --//
    if (isMultiSegment && progress.gpuIndex >= 0) {
      ulong samplesPerSegment = progress.totalSamples / progress.totalGPUs;
      float segmentPercent = static_cast<float>(progress.currentSample) / samplesPerSegment;
      segmentPercent = std::min(1.0f, std::max(0.0f, segmentPercent));
      this->updateSegmentProgress(progress.gpuIndex, segmentPercent);
    }

    //-- Accumulate running loss. --//
    if (!isEpochComplete) {
      this->runningLossSum += static_cast<double>(progress.sampleLoss);
      this->runningLossCount++;
    }

    //-- Throttle output. --//
    if (!isEpochComplete && this->progressReports > 0) {
      ulong interval = std::max(static_cast<ulong>(1), progress.totalSamples / this->progressReports);

      if (progress.currentSample % interval != 0 && progress.currentSample != progress.totalSamples)
        return;
    } else if (!isEpochComplete && this->progressReports == 0) {
      return;
    }

    //-- Compute common values. --//
    double runningAvg = this->runningLossCount > 0 ? this->runningLossSum / this->runningLossCount : 0.0;

    double fractionDone;

    if (isMultiSegment) {
      std::vector<float> segProg = this->getSegmentProgress();
      double sum = 0.0;

      for (float p : segProg)
        sum += p;

      fractionDone = segProg.empty() ? 0.0 : sum / static_cast<double>(segProg.size());
    } else {
      fractionDone =
        (progress.totalSamples > 0) ? static_cast<double>(progress.currentSample) / progress.totalSamples : 0.0;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - this->epochStartTime).count();
    double samplesDone = fractionDone * static_cast<double>(progress.totalSamples);

    //-- Record rate sample for sliding-window calculation (skip epoch-complete frames). --//
    if (this->windowSize > 0 && !isEpochComplete) {
      this->rateWindow.push_back({samplesDone, now});

      if (this->rateWindow.size() > this->windowSize)
        this->rateWindow.pop_front();
    }

    //-- Compute rate using sliding window (or full-epoch fallback). --//
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

    //-- ncurses rendering --//
    {
      std::ostringstream labelStream;
      labelStream << "Epoch " << std::setw(4) << progress.currentEpoch << "/" << progress.totalEpochs;
      std::string label = labelStream.str();

      // Multi-segment: always use renderMultiBar so brackets stay aligned across all frames
      // (epoch-complete uses all-100% segments so the bar looks full).
      if (isMultiSegment) {
        if (isEpochComplete) {
          std::vector<float> allDone(progress.totalGPUs, 1.0f);
          this->renderer.renderMultiBar(win, label, allDone);
        } else {
          std::vector<float> segProg = this->getSegmentProgress();
          this->renderer.renderMultiBar(win, label, segProg);
        }
      } else {
        float samplePercent =
          isEpochComplete ? 1.0f : static_cast<float>(progress.currentSample) / progress.totalSamples;
        this->renderer.renderSingleBar(win, label, samplePercent);
      }

      if (isEpochComplete) {
        std::ostringstream lossText;
        lossText << "Loss: " << std::fixed << std::setprecision(6) << progress.epochLoss << " ";
        this->renderer.renderSubLine(win, lossText.str(), kLossColor);
      } else {
        std::ostringstream statsText;
        statsText << "Loss: " << std::fixed << std::setprecision(6) << runningAvg << "  " << std::setw(6)
                  << static_cast<long>(rate) << " img/s  ETA " << formatEta(eta) << "   ";
        this->renderer.renderSubLine(win, statsText.str());
      }
    }
  }

  //===================================================================================================================//
  //-- Internal Methods --//
  //===================================================================================================================//

  void TrainingProgressTracker::resetSegmentState(int numSegments, ulong epoch)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->totalSegments = numSegments;
    this->currentEpoch = epoch;
    this->segmentProgress.assign(numSegments, 0.0f);
  }

  //===================================================================================================================//

  void TrainingProgressTracker::updateSegmentProgress(int segmentIndex, float percent)
  {
    std::lock_guard<std::mutex> lock(this->mutex);

    if (segmentIndex >= 0 && segmentIndex < static_cast<int>(this->segmentProgress.size()))
      this->segmentProgress[segmentIndex] = percent;
  }

  //===================================================================================================================//

  std::vector<float> TrainingProgressTracker::getSegmentProgress()
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    return this->segmentProgress;
  }

} // namespace NN_CLI
