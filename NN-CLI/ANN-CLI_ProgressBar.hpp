#ifndef ANN_CLI_PROGRESSBAR_HPP
#define ANN_CLI_PROGRESSBAR_HPP

#include <ANN_Core.hpp>

#include <mutex>
#include <vector>

namespace ANN_CLI {

class ProgressBar {
  public:
    ProgressBar(int barWidth = 50);

    // Update and display progress (call from training callback)
    void update(const ANN::TrainingProgress<float>& progress);

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

    bool shouldPrint(const ANN::TrainingProgress<float>& progress, bool isEpochComplete);
};

}  // namespace ANN_CLI

#endif  // ANN_CLI_PROGRESSBAR_HPP

