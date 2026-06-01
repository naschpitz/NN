#ifndef NN_CLI_PROGRESSBAR_HPP
#define NN_CLI_PROGRESSBAR_HPP

#include <chrono>
#include <mutex>
#include <ostream>
#include <vector>

#include <sys/types.h>

namespace NN_CLI
{

  // Common progress info struct used by both ANN and CNN training callbacks
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

  class ProgressBar
  {
    public:
      ProgressBar(ulong progressReports = 1000, int barWidth = 50);

      // Update and display progress (call from training callback)
      void update(const ProgressInfo& progress);

      // When true, the epoch-complete line is NOT terminated with a newline,
      // allowing the caller to append validation loss before ending the line.
      void setHoldEpochLine(bool hold)
      {
        this->holdEpochLine = hold;
      }

      // Reset state (call before starting a new training session)
      void reset();

      // Simple loading progress bar (static, self-contained)
      // Prints: "Loading samples: [████████░░░░░░░░] 1234/5000  24.7%"
      // progressReports controls frequency: how many updates to show (same as trainingConfig.progressReports).
      // Always prints first and last item.
      static void printLoadingProgress(const std::string& label, size_t current, size_t total,
                                       ulong progressReports = 1000, int barWidth = 40);

    private:
      //-- Configuration --//
      ulong progressReports;
      int barWidth;
      bool holdEpochLine = false;

      //-- Per-epoch throughput timer (images/second + ETA) --//
      ulong timerEpoch = 0; // currentEpoch the timer was last reset for
      std::chrono::steady_clock::time_point epochStartTime;

      //-- Multi-GPU progress tracking --//
      std::mutex mutex;
      std::vector<float> gpuProgress; // Progress percentage for each GPU (0.0 - 1.0)
      int totalGPUs = 0;
      ulong currentEpoch = 0;

      //-- Internal methods --//
      void resetGpuState(int numGPUs, ulong epoch);
      void updateGpuProgress(int gpuIndex, float percent);
      std::vector<float> getGpuProgress();

      void renderSingleBar(std::ostream& out, float percent);
      void renderMultiGpuBar(std::ostream& out, const std::vector<float>& gpuProgress, int numGPUs);
  };

} // namespace NN_CLI

#endif // NN_CLI_PROGRESSBAR_HPP
