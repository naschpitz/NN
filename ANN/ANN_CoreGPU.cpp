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
  OpenCLWrapper::Core::initialize(this->logLevel >= LogLevel::DEBUG);

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

  // Adjust batch size to be divisible by numGPUs (round down, minimum = numGPUs)
  ulong batchSize = this->trainingConfig.batchSize;
  batchSize = std::max(this->numGPUs, (batchSize / this->numGPUs) * this->numGPUs);

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Starting GPU training: " << numSamples << " samples, "
              << numEpochs << " epochs, " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "\n";
  }

  struct GPUWorkItem {
    size_t gpuIdx;
    ulong startIdx;
    ulong endIdx;
  };

  // Per-GPU cumulative sample counters for progress tracking across mini-batches
  std::vector<ulong> gpuCumulativeSamples(this->numGPUs, 0);

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = 0;

    // Reset cumulative counters at the start of each epoch
    std::fill(gpuCumulativeSamples.begin(), gpuCumulativeSamples.end(), 0);

    // Process samples in mini-batches
    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      // Distribute the batch across GPUs
      ulong samplesPerGPU = currentBatchSize / this->numGPUs;
      ulong remainder = currentBatchSize % this->numGPUs;

      QVector<GPUWorkItem> workItems;

      for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
        ulong gpuStart = batchStart + gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
        ulong gpuEnd = gpuStart + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
        workItems.append({gpuIdx, gpuStart, gpuEnd});
      }

      std::vector<T> gpuLosses(this->numGPUs, 0);

      // Use QtConcurrent to process each GPU's work in parallel
      QtConcurrent::blockingMap(workItems,
          [this, &samples, &gpuLosses, e, numEpochs, numSamples, &gpuCumulativeSamples](const GPUWorkItem& item) {
        // Create per-batch callback that translates global sample indices to cumulative per-GPU counts
        TrainingCallback<T> callback;
        if (this->trainingCallback) {
          ulong offset = gpuCumulativeSamples[item.gpuIdx];
          ulong batchStartIdx = item.startIdx;
          size_t gpuIdx = item.gpuIdx;
          callback = [this, offset, batchStartIdx, gpuIdx, numSamples](const TrainingProgress<T>& progress) {
            TrainingProgress<T> gpuProgress = progress;
            gpuProgress.currentSample = offset + (progress.currentSample - batchStartIdx);
            gpuProgress.totalSamples = numSamples;
            gpuProgress.gpuIndex = static_cast<int>(gpuIdx);
            gpuProgress.totalGPUs = static_cast<int>(this->numGPUs);
            this->trainingCallback(gpuProgress);
          };
        }
        gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->trainSubset(
            samples, item.startIdx, item.endIdx, e + 1, numEpochs, callback);
      });

      // Update cumulative counters after batch completes
      for (const auto& item : workItems) {
        gpuCumulativeSamples[item.gpuIdx] += (item.endIdx - item.startIdx);
      }

      // Sum up losses from all GPUs for this batch
      for (size_t i = 0; i < this->numGPUs; i++) {
        epochLoss += gpuLosses[i];
      }

      // Merge gradients from all workers and distribute back to all
      this->mergeGradients();

      // Update weights after each mini-batch
      this->update(currentBatchSize);
    }

    // Sync parameters from GPU so getParameters() returns current values (e.g., for checkpoint saves)
    this->gpuWorkers[0]->syncParametersFromGPU();
    this->parameters = this->gpuWorkers[0]->getParameters();

    // Calculate average epoch loss
    T avgEpochLoss = epochLoss / static_cast<T>(numSamples);

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

  std::vector<std::pair<T, ulong>> gpuResults(this->numGPUs, {0, 0});

  // Use QtConcurrent to process each GPU's work in parallel
  QtConcurrent::blockingMap(workItems, [this, &samples, &gpuResults](const GPUWorkItem& item) {
    gpuResults[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(samples, item.startIdx, item.endIdx);
  });

  // Sum up losses and correct counts from all GPUs
  T totalLoss = 0;
  ulong totalCorrect = 0;

  for (size_t i = 0; i < this->numGPUs; i++) {
    totalLoss += gpuResults[i].first;
    totalCorrect += gpuResults[i].second;
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.averageLoss = totalLoss / static_cast<T>(numSamples);
  result.numCorrect = totalCorrect;
  result.accuracy = static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100);

  return result;
}

//===================================================================================================================//
//-- Initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::initializeWorkers() {
  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Initializing GPU training with " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "...\n";
  }

  // Create CoreGPUWorker instances - each will get assigned to a different GPU
  // via OpenCLWrapper's automatic device load balancing
  for (size_t i = 0; i < this->numGPUs; i++) {
    auto worker = std::make_unique<CoreGPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters, this->costFunctionConfig, this->progressReports, this->logLevel);
    this->gpuWorkers.push_back(std::move(worker));
  }

  if (this->logLevel >= LogLevel::INFO) {
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
