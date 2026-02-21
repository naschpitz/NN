#ifndef NN_CLI_PROGRESSBAR_HPP
#define NN_CLI_PROGRESSBAR_HPP

#include <mutex>
#include <ostream>
#include <vector>

#include <sys/types.h>

namespace NN_CLI {

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

class ProgressBar {
  public:
    ProgressBar(int barWidth = 50);

    // Update and display progress (call from training callback)
    void update(const ProgressInfo& progress);

    // Reset state (call before starting a new training session)
    void reset();

  private:
    //-- Configuration --//
    int barWidth;

    //-- Multi-GPU progress tracking --//
    std::mutex mutex;
    std::vector<float> gpuProgress;  // Progress percentage for each GPU (0.0 - 1.0)
    int totalGPUs = 0;
    ulong currentEpoch = 0;

    //-- Output throttling --//
    ulong lastPrintedSample = 0;
    ulong lastPrintedEpoch = 0;

    //-- Internal methods --//
    void resetGpuState(int numGPUs, ulong epoch);
    void updateGpuProgress(int gpuIndex, float percent);
    std::vector<float> getGpuProgress();

    void renderSingleBar(std::ostream& out, float percent);
    void renderMultiGpuBar(std::ostream& out, const std::vector<float>& gpuProgress, int numGPUs);

    bool shouldPrint(const ProgressInfo& progress, bool isEpochComplete);
};

}  // namespace NN_CLI

#endif  // NN_CLI_PROGRESSBAR_HPP

