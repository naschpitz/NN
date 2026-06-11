#ifndef NN_CLI_TRAININGPROGRESSTRACKER_HPP
#define NN_CLI_TRAININGPROGRESSTRACKER_HPP

#include "NN-CLI_TerminalUI_ProgressBar.hpp"
#include "NN-CLI_Types.hpp"

#include <chrono>
#include <deque>
#include <mutex>
#include <vector>

namespace NN_CLI
{

  //-- Progress info struct used by both ANN and CNN training callbacks. --//
  struct ProgressInfo {
      ulong currentEpoch;
      ulong totalEpochs;
      ulong currentSample;
      ulong totalSamples;
      float epochLoss;
      float sampleLoss;
      int gpuIndex;
      int totalGPUs;
  };

  //-- Handles training-specific state and logic, using TerminalUI_ProgressBar for ncurses rendering. --//
  class TrainingProgressTracker
  {
    public:
      //-- Ctors --//
      TrainingProgressTracker(ulong progressReports = 1000, ulong windowSize = 0, int barWidth = 50);

      //-- Public API --//
      void update(const ProgressInfo& progress, WINDOW* win);

    private:
      //-- Types --//
      struct RateSample {
          double samplesProcessed;
          std::chrono::steady_clock::time_point timestamp;
      };

      //-- Internal Methods --//
      void resetSegmentState(int numSegments, ulong epoch);
      void updateSegmentProgress(int segmentIndex, float percent);
      std::vector<float> getSegmentProgress();

      //-- Configuration --//
      ulong progressReports;
      int barWidth;

      //-- Per-epoch throughput timer --//
      ulong timerEpoch = 0;
      std::chrono::steady_clock::time_point epochStartTime;

      //-- Sliding-window throughput measurement --//
      ulong windowSize = 0;
      std::deque<RateSample> rateWindow;

      //-- Running loss accumulator --//
      double runningLossSum = 0.0;
      ulong runningLossCount = 0;

      //-- Multi-segment progress tracking --//
      std::mutex mutex;
      std::vector<float> segmentProgress;
      int totalSegments = 0;
      ulong currentEpoch = 0;

      //-- Renderer --//
      TerminalUI_ProgressBar renderer;
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGPROGRESSTRACKER_HPP
