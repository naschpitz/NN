#include "ANN_CoreGPU.hpp"
#include "ANN_Utils.hpp"

#include <chrono>
#include <iostream>

#include <OCLW_Core.hpp>
#include <QtConcurrent>

using namespace ANN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreGPU<T>::CoreGPU(const CoreConfig<T>& coreConfig)
    : Core<T>(coreConfig) {

  // Initialize OpenCL before querying device information
  OpenCLWrapper::Core::initialize(this->verbose);

  // Determine number of GPUs to use
  int requestedGPUs = coreConfig.trainingConfig.numGPUs;
  size_t availableGPUs = OpenCLWrapper::Core::getDevicesUsage().size();

  if (requestedGPUs == 0) {
    // 0 means use all available GPUs
    this->numGPUs = availableGPUs;
  } else {
    this->numGPUs = std::min(static_cast<size_t>(requestedGPUs), availableGPUs);
  }

  // Initialize GPU workers
  this->initializeWorkers();
}
//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPU<T>::predict(const Input<T>& input) {
  this->predictStart();

  // Delegate to the first worker
  Output<T> output = this->gpuWorkers[0]->predict(input);

  this->predictEnd();

  return output;
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::train(const Samples<T>& samples) {
  this->trainingStart(samples.size());

  ulong numSamples = samples.size();
  ulong numEpochs = this->trainingConfig.numEpochs;

  if (this->verbose) {
    std::cout << "Starting GPU training: " << numSamples << " samples, "
              << numEpochs << " epochs, " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "\n";
  }

  // Create per-GPU wrapper callbacks that inject GPU index into progress reports
  auto createGpuCallback = [this](size_t gpuIdx) -> TrainingCallback<T> {
    if (!this->trainingCallback) return nullptr;

    return [this, gpuIdx](const TrainingProgress<T>& progress) {
      TrainingProgress<T> gpuProgress = progress;
      gpuProgress.gpuIndex = static_cast<int>(gpuIdx);
      gpuProgress.totalGPUs = static_cast<int>(this->numGPUs);
      this->trainingCallback(gpuProgress);
    };
  };

  for (ulong e = 0; e < numEpochs; e++) {
    // Calculate sample ranges for each GPU
    ulong samplesPerGPU = numSamples / this->numGPUs;
    ulong remainder = numSamples % this->numGPUs;

    // Build list of GPU indices and their sample ranges
    struct GPUWorkItem {
      size_t gpuIdx;
      ulong startIdx;
      ulong endIdx;
    };

    QVector<GPUWorkItem> workItems;

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
      ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
      workItems.append({gpuIdx, startIdx, endIdx});
    }

    std::vector<T> gpuLosses(this->numGPUs, 0);

    // Use QtConcurrent to process each GPU's work in parallel
    QtConcurrent::blockingMap(workItems, [this, &samples, &gpuLosses, e, numEpochs, &createGpuCallback](const GPUWorkItem& item) {
      gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->trainSubset(
          samples, item.startIdx, item.endIdx, e + 1, numEpochs, createGpuCallback(item.gpuIdx));
    });

    // Sum up losses from all GPUs
    T totalLoss = 0;

    for (size_t i = 0; i < this->numGPUs; i++) {
      totalLoss += gpuLosses[i];
    }

    // Merge gradients from all workers and distribute back to all
    this->mergeGradients();

    // Update weights on all workers
    this->update(numSamples);

    // Calculate average epoch loss
    T avgEpochLoss = totalLoss / static_cast<T>(numSamples);

    // Store final loss from the last epoch
    this->trainingMetadata.finalLoss = avgEpochLoss;

    // Report epoch completion (gpuIndex = -1 indicates combined result)
    if (this->trainingCallback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = e + 1;
      progress.totalEpochs = numEpochs;
      progress.currentSample = numSamples;
      progress.totalSamples = numSamples;
      progress.sampleLoss = 0;
      progress.epochLoss = avgEpochLoss;
      progress.gpuIndex = -1;  // Epoch completion is not GPU-specific
      progress.totalGPUs = static_cast<int>(this->numGPUs);
      this->trainingCallback(progress);
    }
  }

  this->trainingEnd();

  // Sync final parameters from any worker (all have the same weights now)
  this->gpuWorkers[0]->syncParametersFromGPU();
  this->parameters = this->gpuWorkers[0]->getParameters();
}

//===================================================================================================================//
//-- Testing --//
//===================================================================================================================//

template <typename T>
TestResult<T> CoreGPU<T>::test(const Samples<T>& samples) {
  ulong numSamples = samples.size();

  // Calculate sample ranges for each GPU
  ulong samplesPerGPU = numSamples / this->numGPUs;
  ulong remainder = numSamples % this->numGPUs;

  // Build list of GPU indices and their sample ranges
  struct GPUWorkItem {
    size_t gpuIdx;
    ulong startIdx;
    ulong endIdx;
  };

  QVector<GPUWorkItem> workItems;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
    ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
    workItems.append({gpuIdx, startIdx, endIdx});
  }

  std::vector<T> gpuLosses(this->numGPUs, 0);

  // Use QtConcurrent to process each GPU's work in parallel
  QtConcurrent::blockingMap(workItems, [this, &samples, &gpuLosses](const GPUWorkItem& item) {
    gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(samples, item.startIdx, item.endIdx);
  });

  // Sum up losses from all GPUs
  T totalLoss = 0;

  for (size_t i = 0; i < this->numGPUs; i++) {
    totalLoss += gpuLosses[i];
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.averageLoss = totalLoss / static_cast<T>(numSamples);

  return result;
}

//===================================================================================================================//
//-- Initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::initializeWorkers() {
  if (this->verbose) {
    std::cout << "Initializing GPU training with " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "...\n";
  }

  // Create CoreGPUWorker instances - each will get assigned to a different GPU
  // via OpenCLWrapper's automatic device load balancing
  for (size_t i = 0; i < this->numGPUs; i++) {
    auto worker = std::make_unique<CoreGPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters, this->progressReports, this->verbose);
    this->gpuWorkers.push_back(std::move(worker));
  }

  if (this->verbose) {
    std::cout << "GPU initialization complete.\n";
  }
}

//===================================================================================================================//
//-- Multi-GPU coordination --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeGradients() {
  // Read and sum gradients from ALL workers
  Tensor1D<T> totalAccumWeights;
  Tensor1D<T> totalAccumBiases;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    Tensor1D<T> workerWeights;
    Tensor1D<T> workerBiases;

    this->gpuWorkers[gpuIdx]->readAccumulatedGradients(workerWeights, workerBiases);

    if (gpuIdx == 0) {
      // First worker - initialize the totals
      totalAccumWeights = workerWeights;
      totalAccumBiases = workerBiases;
    } else {
      // Add to totals
      for (size_t i = 0; i < workerWeights.size(); i++) {
        totalAccumWeights[i] += workerWeights[i];
      }

      for (size_t i = 0; i < workerBiases.size(); i++) {
        totalAccumBiases[i] += workerBiases[i];
      }
    }
  }

  // Write merged gradients back to ALL workers
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->setAccumulators(totalAccumWeights, totalAccumBiases);
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::update(ulong numSamples) {
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->update(numSamples);
  }
}

//===================================================================================================================//
//-- Step-by-step training methods (for external orchestration, e.g., CNN) --//
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreGPU<T>::backpropagate(const Output<T>& output) {
  // Delegate to the first worker (same as predict)
  return this->gpuWorkers[0]->backpropagate(output);
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::accumulate() {
  // Delegate to the first worker (same as predict)
  this->gpuWorkers[0]->accumulate();
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::resetAccumulators() {
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->resetAccumulators();
  }
}

//===================================================================================================================//
// (Optional) Explicit template instantiations.
template class ANN::CoreGPU<int>;
template class ANN::CoreGPU<double>;
template class ANN::CoreGPU<float>;
