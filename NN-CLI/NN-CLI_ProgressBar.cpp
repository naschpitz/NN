#include "NN-CLI_ProgressBar.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace NN_CLI {

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

ProgressBar::ProgressBar(ulong progressReports, int barWidth)
    : progressReports(progressReports), barWidth(barWidth) {}

//===================================================================================================================//
//-- Public Interface --//
//===================================================================================================================//

void ProgressBar::update(const ProgressInfo& progress) {
  bool isEpochComplete = (progress.epochLoss > 0);
  bool isMultiGPU = (progress.totalGPUs > 1);

  // Reset GPU state at the start of each epoch
  if (progress.gpuIndex >= 0 && this->currentEpoch != progress.currentEpoch) {
    this->resetGpuState(progress.totalGPUs, progress.currentEpoch);
  }

  // For multi-GPU, update per-GPU progress
  if (isMultiGPU && progress.gpuIndex >= 0) {
    // Calculate this GPU's progress within its assigned subset
    ulong samplesPerGPU = progress.totalSamples / progress.totalGPUs;
    ulong gpuStartIdx = progress.gpuIndex * samplesPerGPU;
    ulong gpuSamplesProcessed = progress.currentSample - gpuStartIdx;
    float gpuPercent = static_cast<float>(gpuSamplesProcessed) / samplesPerGPU;
    gpuPercent = std::min(1.0f, std::max(0.0f, gpuPercent));

    this->updateGpuProgress(progress.gpuIndex, gpuPercent);
  }

  // Throttle output based on progressReports
  if (!isEpochComplete && this->progressReports > 0) {
    ulong interval = std::max(static_cast<ulong>(1), progress.totalSamples / this->progressReports);
    if (progress.currentSample % interval != 0 && progress.currentSample != progress.totalSamples) {
      return;
    }
  } else if (!isEpochComplete && this->progressReports == 0) {
    return;  // Suppress all sample progress output
  }

  // Build output
  std::ostringstream out;
  out << "\rEpoch " << std::setw(4) << progress.currentEpoch << "/" << progress.totalEpochs << " [";

  if (isMultiGPU && !isEpochComplete) {
    std::vector<float> gpuProg = this->getGpuProgress();
    this->renderMultiGpuBar(out, gpuProg, progress.totalGPUs);
  } else {
    float samplePercent = static_cast<float>(progress.currentSample) / progress.totalSamples;
    this->renderSingleBar(out, samplePercent);
  }

  // Show loss information
  if (isEpochComplete) {
    out << " - Loss: " << std::fixed << std::setprecision(6) << progress.epochLoss;
    out << std::string(30, ' ') << std::endl;
  } else {
    out << " - Loss: " << std::fixed << std::setprecision(6) << progress.sampleLoss << "   ";
  }

  std::cout << out.str() << std::flush;
}

void ProgressBar::reset() {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->gpuProgress.clear();
  this->totalGPUs = 0;
  this->currentEpoch = 0;
}

//===================================================================================================================//
//-- Loading Progress --//
//===================================================================================================================//

void ProgressBar::printLoadingProgress(const std::string& label, size_t current, size_t total,
                                        ulong progressReports, int barWidth) {
  // Compute reporting interval from progressReports (number of reports desired)
  ulong interval = (progressReports > 0) ? std::max(static_cast<size_t>(1), total / progressReports) : 0;

  // If progressReports is 0, suppress all output
  if (interval == 0) return;

  // Throttle: only print at first, last, and every interval items
  if (current != 1 && current != total && (current % interval) != 0) {
    return;
  }

  float percent = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
  int filledWidth = static_cast<int>(percent * barWidth);

  std::ostringstream out;
  out << "\r" << label << " [";

  for (int i = 0; i < barWidth; i++) {
    out << (i < filledWidth ? "█" : "░");
  }

  out << "] " << current << "/" << total
      << "  " << std::fixed << std::setprecision(1) << (percent * 100.0f) << "%";

  // Pad with spaces to clear any leftover characters from previous longer lines
  out << "   ";

  std::cout << out.str() << std::flush;

  // Print newline when complete
  if (current == total) {
    std::cout << std::endl;
  }
}

//===================================================================================================================//
//-- GPU State Management --//
//===================================================================================================================//

void ProgressBar::resetGpuState(int numGPUs, ulong epoch) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->totalGPUs = numGPUs;
  this->currentEpoch = epoch;
  this->gpuProgress.assign(numGPUs, 0.0f);
}

void ProgressBar::updateGpuProgress(int gpuIndex, float percent) {
  std::lock_guard<std::mutex> lock(this->mutex);
  if (gpuIndex >= 0 && gpuIndex < static_cast<int>(this->gpuProgress.size())) {
    this->gpuProgress[gpuIndex] = percent;
  }
}

std::vector<float> ProgressBar::getGpuProgress() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->gpuProgress;
}

//===================================================================================================================//
//-- Rendering --//
//===================================================================================================================//

void ProgressBar::renderSingleBar(std::ostream& out, float percent) {
  int filledWidth = static_cast<int>(percent * this->barWidth);

  for (int i = 0; i < this->barWidth; i++) {
    out << (i < filledWidth ? "█" : "░");
  }

  out << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (percent * 100) << "%";
}

void ProgressBar::renderMultiGpuBar(std::ostream& out, const std::vector<float>& gpuProg, int numGPUs) {
  int segmentWidth = this->barWidth / numGPUs;

  for (int gpu = 0; gpu < numGPUs; gpu++) {
    float gpuPercent = (gpu < static_cast<int>(gpuProg.size())) ? gpuProg[gpu] : 0.0f;
    int filledWidth = static_cast<int>(gpuPercent * segmentWidth);

    for (int i = 0; i < segmentWidth; i++) {
      out << (i < filledWidth ? "█" : "░");
    }

    // Add separator between GPU segments (except after last)
    if (gpu < numGPUs - 1) {
      out << "│";
    }
  }

  // Calculate average progress
  float totalPercent = 0.0f;
  for (float p : gpuProg) {
    totalPercent += p;
  }
  if (numGPUs > 0) {
    totalPercent /= numGPUs;
  }

  out << "] " << std::fixed << std::setprecision(1) << std::setw(5) << (totalPercent * 100) << "% ";

  // Show per-GPU percentages
  out << "(";
  for (int gpu = 0; gpu < numGPUs; gpu++) {
    float gpuPercent = (gpu < static_cast<int>(gpuProg.size())) ? gpuProg[gpu] : 0.0f;
    out << gpu << ":" << std::setw(3) << static_cast<int>(gpuPercent * 100) << "%";
    if (gpu < numGPUs - 1) out << " | ";
  }
  out << ")";
}

}  // namespace NN_CLI

