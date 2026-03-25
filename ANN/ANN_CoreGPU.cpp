#include "ANN_CoreGPU.hpp"
#include "ANN_Utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

#include <OCLW_Core.hpp>
#include <QtConcurrent>

using namespace ANN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreGPU<T>::CoreGPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig)
{
  // Initialize OpenCL before querying device information
  OpenCLWrapper::Core::initialize(this->logLevel >= LogLevel::DEBUG);

  // Determine number of GPUs to use
  int requestedGPUs = coreConfig.numGPUs;
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
Outputs<T> CoreGPU<T>::predict(const Inputs<T>& inputs)
{
  this->predictStart();

  ulong numInputs = inputs.size();
  Outputs<T> outputs(numInputs);

  // Distribute inputs across GPUs
  ulong inputsPerGPU = numInputs / this->numGPUs;
  ulong remainder = numInputs % this->numGPUs;

  struct GPUWorkItem {
      size_t gpuIdx;
      ulong startIdx;
      ulong endIdx;
  };

  QVector<GPUWorkItem> workItems;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    ulong startIdx = gpuIdx * inputsPerGPU + std::min(gpuIdx, remainder);
    ulong endIdx = startIdx + inputsPerGPU + (gpuIdx < remainder ? 1 : 0);

    if (startIdx < endIdx)
      workItems.append({gpuIdx, startIdx, endIdx});
  }

  std::vector<Outputs<T>> gpuOutputs(this->numGPUs);
  std::atomic<ulong> completedInputs{0};

  QtConcurrent::blockingMap(
    workItems, [this, &inputs, &gpuOutputs, &completedInputs, numInputs](const GPUWorkItem& item) {
      ProgressCallback callback;

      if (this->progressCallback) {
        callback = [this, &completedInputs, numInputs](ulong /*current*/, ulong /*total*/) {
          ulong completed = ++completedInputs;
          this->progressCallback(completed, numInputs);
        };
      }

      gpuOutputs[item.gpuIdx] =
        this->gpuWorkers[item.gpuIdx]->predictSubset(inputs, item.startIdx, item.endIdx, callback);
    });

  // Merge GPU outputs into final result
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    ulong startIdx = gpuIdx * inputsPerGPU + std::min(gpuIdx, remainder);

    for (ulong i = 0; i < gpuOutputs[gpuIdx].size(); i++)
      outputs[startIdx + i] = std::move(gpuOutputs[gpuIdx][i]);
  }

  this->predictEnd();

  return outputs;
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  this->trainingStart(numSamples);

  ulong numEpochs = this->trainingConfig.numEpochs;

  // Adjust batch size to be divisible by numGPUs (round down, minimum = numGPUs)
  ulong batchSize = this->trainingConfig.batchSize;
  batchSize = std::max(this->numGPUs, (batchSize / this->numGPUs) * this->numGPUs);

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Starting GPU training: " << numSamples << " samples, " << numEpochs << " epochs, " << this->numGPUs
              << " GPU" << (this->numGPUs > 1 ? "s" : "") << "\n";
  }

  struct GPUWorkItem {
      size_t gpuIdx;
      ulong localStart; // Start index into batchSamples (0-based)
      ulong localEnd; // End index into batchSamples
  };

  // Per-GPU cumulative sample counters for progress tracking across mini-batches
  std::vector<ulong> gpuCumulativeSamples(this->numGPUs, 0);

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  std::mt19937 rng(std::random_device{}());

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = 0;

    // Shuffle sample order for this epoch
    if (this->trainingConfig.shuffleSamples) {
      std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);
    }

    // Reset cumulative counters at the start of each epoch
    std::fill(gpuCumulativeSamples.begin(), gpuCumulativeSamples.end(), 0);

    // Process samples in mini-batches
    ulong batchIndex = 0;

    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize, batchIndex++) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      // Fetch batch samples via provider
      Samples<T> batchSamples = sampleProvider(sampleIndices, batchSize, batchIndex);

      // Distribute the batch across GPUs (using local 0-based indices into batchSamples)
      ulong samplesPerGPU = currentBatchSize / this->numGPUs;
      ulong remainder = currentBatchSize % this->numGPUs;

      QVector<GPUWorkItem> workItems;

      for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
        ulong localStart = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
        ulong localEnd = localStart + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
        workItems.append({gpuIdx, localStart, localEnd});
      }

      std::vector<T> gpuLosses(this->numGPUs, 0);

      // Use QtConcurrent to process each GPU's work in parallel
      QtConcurrent::blockingMap(workItems, [this, &batchSamples, &gpuLosses, e, numEpochs, numSamples,
                                            &gpuCumulativeSamples](const GPUWorkItem& item) {
        // Build the per-GPU sub-batch
        Samples<T> gpuSamples(batchSamples.begin() + item.localStart, batchSamples.begin() + item.localEnd);

        // Create per-batch callback that translates local indices to cumulative per-GPU counts
        TrainingCallback<T> callback;

        if (this->trainingCallback) {
          ulong offset = gpuCumulativeSamples[item.gpuIdx];
          size_t gpuIdx = item.gpuIdx;
          callback = [this, offset, gpuIdx, numSamples](const TrainingProgress<T>& progress) {
            TrainingProgress<T> gpuProgress = progress;
            gpuProgress.currentSample = offset + progress.currentSample;
            gpuProgress.totalSamples = numSamples;
            gpuProgress.gpuIndex = static_cast<int>(gpuIdx);
            gpuProgress.totalGPUs = static_cast<int>(this->numGPUs);
            this->trainingCallback(gpuProgress);
          };
        }

        gpuLosses[item.gpuIdx] =
          this->gpuWorkers[item.gpuIdx]->trainSubset(gpuSamples, numSamples, e + 1, numEpochs, callback);
      });

      // Update cumulative counters after batch completes
      for (const auto& item : workItems) {
        gpuCumulativeSamples[item.gpuIdx] += (item.localEnd - item.localStart);
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
    this->gpuWorkers[0]->bufferManager->syncParametersFromGPU();
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
      progress.gpuIndex = -1; // Epoch completion is not GPU-specific
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
TestResult<T> CoreGPU<T>::test(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  // Sequential index array (no shuffling for test)
  std::vector<ulong> sampleIndices(numSamples);

  for (ulong i = 0; i < numSamples; i++) {
    sampleIndices[i] = i;
  }

  ulong batchSize = this->testConfig.batchSize;
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;

  T totalLoss = static_cast<T>(0);
  ulong totalCorrect = 0;

  for (ulong b = 0; b < numBatches; b++) {
    Samples<T> batch = sampleProvider(sampleIndices, batchSize, b);

    // Distribute batch across GPUs
    ulong batchLen = batch.size();
    ulong samplesPerGPU = batchLen / this->numGPUs;
    ulong remainder = batchLen % this->numGPUs;

    struct GPUWorkItem {
        size_t gpuIdx;
        ulong startIdx;
        ulong endIdx;
    };

    QVector<GPUWorkItem> workItems;

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
      ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);

      if (startIdx < endIdx)
        workItems.append({gpuIdx, startIdx, endIdx});
    }

    std::vector<std::pair<T, ulong>> gpuResults(this->numGPUs, {0, 0});

    QtConcurrent::blockingMap(workItems, [this, &batch, &gpuResults](const GPUWorkItem& item) {
      gpuResults[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(batch, item.startIdx, item.endIdx);
    });

    for (size_t i = 0; i < this->numGPUs; i++) {
      totalLoss += gpuResults[i].first;
      totalCorrect += gpuResults[i].second;
    }

    if (this->progressCallback) {
      ulong samplesProcessed = std::min((b + 1) * batchSize, numSamples);
      this->progressCallback(samplesProcessed, numSamples);
    }
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.numCorrect = totalCorrect;
  result.averageLoss = (numSamples > 0) ? totalLoss / static_cast<T>(numSamples) : static_cast<T>(0);
  result.accuracy = (numSamples > 0) ? static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100)
                                     : static_cast<T>(0);

  return result;
}

//===================================================================================================================//
//-- Initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::initializeWorkers()
{
  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Initializing GPU training with " << this->numGPUs << " GPU" << (this->numGPUs > 1 ? "s" : "")
              << "...\n";
  }

  // Create CoreGPUWorker instances - each will get assigned to a different GPU
  // via OpenCLWrapper's automatic device load balancing
  for (size_t i = 0; i < this->numGPUs; i++) {
    auto worker = std::make_unique<CoreGPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters,
                                                     this->costFunctionConfig, this->progressReports, this->logLevel);
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
void CoreGPU<T>::mergeGradients()
{
  // Read and sum gradients from ALL workers
  Tensor1D<T> totalAccumWeights;
  Tensor1D<T> totalAccumBiases;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    Tensor1D<T> workerWeights;
    Tensor1D<T> workerBiases;

    this->gpuWorkers[gpuIdx]->bufferManager->readAccumulatedGradients(workerWeights, workerBiases);

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
    this->gpuWorkers[gpuIdx]->bufferManager->setAccumulators(totalAccumWeights, totalAccumBiases);
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::update(ulong numSamples)
{
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->update(numSamples);
  }
}

//===================================================================================================================//
//-- Step-by-step training methods (for external orchestration, e.g., CNN) --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::resetAccumulators()
{
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->resetAccumulators();
  }
}

//===================================================================================================================//
// (Optional) Explicit template instantiations.
template class ANN::CoreGPU<int>;
template class ANN::CoreGPU<double>;
template class ANN::CoreGPU<float>;
