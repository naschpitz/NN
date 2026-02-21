#include "NN-CLI_ProgressBar.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace NN_CLI {

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

ProgressBar::ProgressBar(int barWidth) : barWidth(barWidth) {}

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

  // Throttle output to avoid flooding
  if (!this->shouldPrint(progress, isEpochComplete)) {
    return;
  }

  this->lastPrintedSample = progress.currentSample;
  this->lastPrintedEpoch = progress.currentEpoch;

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
  this->lastPrintedSample = 0;
  this->lastPrintedEpoch = 0;
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

bool ProgressBar::shouldPrint(const ProgressInfo& progress, bool isEpochComplete) {
  return isEpochComplete ||
         (progress.currentEpoch != this->lastPrintedEpoch) ||
         (progress.currentSample >= this->lastPrintedSample + 500) ||
         (progress.currentSample == 1);
}

}  // namespace NN_CLI

