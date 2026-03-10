#include "ANN_CoreCPU.hpp"
#include "ANN_Utils.hpp"

#include <QThreadPool>
#include <QtConcurrent>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <random>
#include <cmath>

using namespace ANN;

//===================================================================================================================//

template <typename T>
CoreCPU<T>::CoreCPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig)
{
  this->initializeParameters();

  // Create the step worker (used for predict and step-by-step training path)
  bool allocateTraining = (this->modeType == ModeType::TRAIN);
  this->stepWorker = std::make_unique<CoreCPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters,
                                                        this->costFunctionConfig, allocateTraining);

  // Note: Global accumulators and Adam state are allocated lazily in train(),
  // NOT here. This ensures the step-by-step API (used by CNN) sees empty global
  // accumulators, so update() correctly reads from stepWorker's accumulators.
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::predict(const Input<T>& input)
{
  this->predictStart();

  this->stepWorker->propagate(input);
  Output<T> output = this->stepWorker->getOutput();

  this->predictEnd();

  return output;
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  // Allocate global accumulators (for merging worker gradients) and Adam state on first call.
  // These are NOT allocated in the constructor so that the step-by-step API
  // (predict→backpropagate→accumulate→update) correctly uses stepWorker's accumulators.
  if (this->accum_dCost_dWeights.empty()) {
    this->allocateGlobalAccumulators();

    if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
      this->allocateAdamState();
    }
  }

  this->trainingStart(numSamples);

  ulong numEpochs = this->trainingConfig.numEpochs;

  // Use configured numThreads, or all available cores if 0
  int numThreads = this->numThreads;

  if (numThreads <= 0) {
    numThreads = QThreadPool::globalInstance()->maxThreadCount();
  }

  // Adjust batch size to be divisible by numThreads (round down, minimum = numThreads)
  ulong batchSize = this->trainingConfig.batchSize;
  ulong numWorkers = static_cast<ulong>(numThreads);
  batchSize = std::max(numWorkers, (batchSize / numWorkers) * numWorkers);

  // Pre-allocate workers for each thread
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(std::make_unique<CoreCPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters,
                                                         this->costFunctionConfig, true));
  }

  // Atomic counter for assigning unique worker indices to threads
  std::atomic<int> nextWorkerIndex{0};

  // Mutex for serializing callback calls (prevents I/O contention)
  QMutex callbackMutex;

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

    // Atomic counter for progress tracking
    std::atomic<ulong> completedSamples{0};

    // Process samples in mini-batches
    ulong batchIndex = 0;

    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize, batchIndex++) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      // Reset global accumulators at the start of each batch
      this->resetGlobalAccumulators();

      // Reset worker accumulators at the start of each batch (including loss)
      for (int i = 0; i < numThreads; i++) {
        workers[i]->resetAccumulators();
        workers[i]->resetAccumLoss();
      }

      // Fetch batch samples via provider
      Samples<T> batchSamples = sampleProvider(sampleIndices, batchSize, batchIndex);

      // Use blockingMap to process all samples in the batch in parallel
      QVector<ulong> localIndices(currentBatchSize);
      std::iota(localIndices.begin(), localIndices.end(), 0);

      QtConcurrent::blockingMap(localIndices, [&, numThreads](ulong localIdx) {
        // Each thread gets a unique worker index on first use (thread_local persists for thread lifetime)
        thread_local int workerIndex = nextWorkerIndex.fetch_add(1) % numThreads;
        CoreCPUWorker<T>& worker = *workers[workerIndex];

        // Process the sample
        const Sample<T>& sample = batchSamples[localIdx];
        worker.propagate(sample.input, true);
        T sampleLoss = worker.computeLoss(sample.output);

        worker.backpropagate(sample.output);

        // Accumulate loss to worker's local accumulator (no atomic needed - each thread has its own worker)
        worker.addToAccumLoss(sampleLoss);

        // Accumulate gradients to worker's local accumulators (no mutex needed)
        worker.accumulate();

        // Increment completed samples counter and report progress
        ulong completed = ++completedSamples;
        this->reportProgress(e + 1, numEpochs, completed, numSamples, sampleLoss, 0, callbackMutex);
      });

      // Merge all worker accumulators into global accumulators
      for (int i = 0; i < numThreads; i++) {
        this->mergeWorkerAccumulators(*workers[i]);
        epochLoss += workers[i]->getAccumLoss();
      }

      // Update weights after each mini-batch
      this->update(currentBatchSize);
    }

    // Report epoch completion
    T avgEpochLoss = epochLoss / static_cast<T>(numSamples);
    this->reportProgress(e + 1, numEpochs, numSamples, numSamples, 0, avgEpochLoss, callbackMutex);

    // Store final loss from the last epoch
    this->trainingMetadata.finalLoss = avgEpochLoss;
  }

  this->trainingEnd();
}

//===================================================================================================================//

template <typename T>
TestResult<T> CoreCPU<T>::test(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  // Use configured numThreads, or all available cores if 0
  int numThreads = this->numThreads;

  if (numThreads <= 0) {
    numThreads = QThreadPool::globalInstance()->maxThreadCount();
  }

  // Pre-allocate workers for each thread (forward pass only — no training buffers)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(std::make_unique<CoreCPUWorker<T>>(this->layersConfig, this->trainingConfig, this->parameters,
                                                         this->costFunctionConfig, false));
  }

  ulong numLayers = this->layersConfig.size();

  // Sequential index array (no shuffling for test)
  std::vector<ulong> sampleIndices(numSamples);

  for (ulong i = 0; i < numSamples; i++) {
    sampleIndices[i] = i;
  }

  // Process in batches via the provider
  ulong batchSize = this->testConfig.batchSize;
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;

  T totalLoss = 0;
  ulong totalCorrect = 0;

  for (ulong b = 0; b < numBatches; b++) {
    Samples<T> batch = sampleProvider(sampleIndices, batchSize, b);

    // Atomic counter for assigning unique worker indices to threads
    std::atomic<int> nextWorkerIndex{0};

    // Per-worker loss and correct counters
    std::vector<T> workerLosses(numThreads, 0);
    std::vector<ulong> workerCorrects(numThreads, 0);

    QVector<ulong> batchIndices(batch.size());

    for (ulong i = 0; i < batch.size(); i++) {
      batchIndices[i] = i;
    }

    QtConcurrent::blockingMap(batchIndices, [&, numThreads, numLayers](ulong idx) {
      thread_local int workerIndex = nextWorkerIndex.fetch_add(1) % numThreads;
      CoreCPUWorker<T>& worker = *workers[workerIndex];

      const Sample<T>& sample = batch[idx];
      worker.propagate(sample.input);
      T sampleLoss = worker.computeLoss(sample.output);

      workerLosses[workerIndex] += sampleLoss;

      const auto& outputActvs = worker.getActvs()[numLayers - 1];
      auto predIdx = std::distance(outputActvs.begin(), std::max_element(outputActvs.begin(), outputActvs.end()));
      auto expIdx = std::distance(sample.output.begin(), std::max_element(sample.output.begin(), sample.output.end()));

      if (predIdx == expIdx)
        workerCorrects[workerIndex]++;
    });

    for (int i = 0; i < numThreads; i++) {
      totalLoss += workerLosses[i];
      totalCorrect += workerCorrects[i];
    }

    if (this->progressCallback) {
      ulong samplesProcessed = std::min((b + 1) * batchSize, numSamples);
      this->progressCallback(samplesProcessed, numSamples);
    }
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.averageLoss = (numSamples > 0) ? totalLoss / static_cast<T>(numSamples) : static_cast<T>(0);
  result.numCorrect = totalCorrect;
  result.accuracy = (numSamples > 0) ? static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100)
                                     : static_cast<T>(0);

  return result;
}

//===================================================================================================================//
// Initialization
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::initializeParameters()
{
  ulong numLayers = this->layersConfig.size();

  // Check if parameters were loaded from file (non-empty)
  bool hasLoadedParameters = !this->parameters.weights.empty() && this->parameters.weights.size() == numLayers;

  if (hasLoadedParameters)
    return;

  // Initialize weights and biases
  this->parameters.weights.resize(numLayers);
  this->parameters.biases.resize(numLayers);

  // Random number generator for weight initialization
  std::random_device rd;
  std::mt19937 gen(rd());

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;
    ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

    this->parameters.weights[l].resize(numNeurons);
    this->parameters.biases[l].resize(numNeurons, static_cast<T>(0));

    // He initialization for ReLU, Xavier for sigmoid/tanh
    ActvFuncType actvFuncType = layer.actvFuncType;
    double stddev;

    if (actvFuncType == ActvFuncType::RELU) {
      stddev = std::sqrt(2.0 / static_cast<double>(prevNumNeurons));
    } else {
      stddev = std::sqrt(1.0 / static_cast<double>(prevNumNeurons));
    }

    std::normal_distribution<double> dist(0.0, stddev);

    for (ulong j = 0; j < numNeurons; j++) {
      this->parameters.weights[l][j].resize(prevNumNeurons);

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->parameters.weights[l][j][k] = static_cast<T>(dist(gen));
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateGlobalAccumulators()
{
  ulong numLayers = this->layersConfig.size();

  this->accum_dCost_dWeights.resize(numLayers);
  this->accum_dCost_dBiases.resize(numLayers);

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;
    ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

    this->accum_dCost_dWeights[l].resize(numNeurons);
    this->accum_dCost_dBiases[l].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; j++) {
      this->accum_dCost_dWeights[l][j].resize(prevNumNeurons);
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateAdamState()
{
  ulong numLayers = this->layersConfig.size();

  this->adam_m_weights.resize(numLayers);
  this->adam_m_biases.resize(numLayers);
  this->adam_v_weights.resize(numLayers);
  this->adam_v_biases.resize(numLayers);

  for (ulong l = 1; l < numLayers; l++) {
    ulong numNeurons = this->layersConfig[l].numNeurons;
    ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

    this->adam_m_weights[l].resize(numNeurons);
    this->adam_m_biases[l].resize(numNeurons, static_cast<T>(0));
    this->adam_v_weights[l].resize(numNeurons);
    this->adam_v_biases[l].resize(numNeurons, static_cast<T>(0));

    for (ulong j = 0; j < numNeurons; j++) {
      this->adam_m_weights[l][j].resize(prevNumNeurons, static_cast<T>(0));
      this->adam_v_weights[l][j].resize(prevNumNeurons, static_cast<T>(0));
    }
  }

  this->adam_t = 0;
}

//===================================================================================================================//
// Training helpers
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::resetGlobalAccumulators()
{
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      this->accum_dCost_dBiases[l][j] = static_cast<T>(0);

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->accum_dCost_dWeights[l][j][k] = static_cast<T>(0);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::mergeWorkerAccumulators(const CoreCPUWorker<T>& worker)
{
  QMutexLocker locker(&this->accumulatorMutex);

  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      this->accum_dCost_dBiases[l][j] += worker.getAccumBiases()[l][j];

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->accum_dCost_dWeights[l][j][k] += worker.getAccumWeights()[l][j][k];
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::update(ulong numSamples)
{
  T learningRate = this->trainingConfig.learningRate;
  ulong numLayers = this->layersConfig.size();

  // Determine which accumulators to read from:
  // - train() merges worker accumulators into global accumulators, then calls update()
  // - step-by-step path (CNN) accumulates directly into stepWorker's accumulators
  const Tensor3D<T>& accumWeights =
    this->accum_dCost_dWeights.empty() ? this->stepWorker->getAccumWeights() : this->accum_dCost_dWeights;
  const Tensor2D<T>& accumBiases =
    this->accum_dCost_dBiases.empty() ? this->stepWorker->getAccumBiases() : this->accum_dCost_dBiases;

  T n = static_cast<T>(numSamples);

  if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
    // Lazily allocate ADAM state on first update() call.
    // This handles the step-by-step path (CNN) where train() is never called.
    if (this->adam_m_weights.empty()) {
      this->allocateAdamState();
    }

    const auto& opt = this->trainingConfig.optimizer;
    T beta1 = opt.beta1;
    T beta2 = opt.beta2;
    T epsilon = opt.epsilon;

    this->adam_t++;

    T bc1 = static_cast<T>(1) - std::pow(beta1, static_cast<T>(this->adam_t));
    T bc2 = static_cast<T>(1) - std::pow(beta2, static_cast<T>(this->adam_t));

    for (ulong l = 1; l < numLayers; l++) {
      ulong numNeurons = this->layersConfig[l].numNeurons;
      ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

      for (ulong j = 0; j < numNeurons; j++) {
        T g = accumBiases[l][j] / n;
        this->adam_m_biases[l][j] = beta1 * this->adam_m_biases[l][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_biases[l][j] = beta2 * this->adam_v_biases[l][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_biases[l][j] / bc1;
        T v_hat = this->adam_v_biases[l][j] / bc2;
        this->parameters.biases[l][j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);

        for (ulong k = 0; k < prevNumNeurons; k++) {
          g = accumWeights[l][j][k] / n;
          this->adam_m_weights[l][j][k] = beta1 * this->adam_m_weights[l][j][k] + (static_cast<T>(1) - beta1) * g;
          this->adam_v_weights[l][j][k] = beta2 * this->adam_v_weights[l][j][k] + (static_cast<T>(1) - beta2) * g * g;
          m_hat = this->adam_m_weights[l][j][k] / bc1;
          v_hat = this->adam_v_weights[l][j][k] / bc2;
          this->parameters.weights[l][j][k] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
      }
    }
  } else {
    // SGD
    for (ulong l = 1; l < numLayers; l++) {
      ulong numNeurons = this->layersConfig[l].numNeurons;
      ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

      for (ulong j = 0; j < numNeurons; j++) {
        this->parameters.biases[l][j] -= learningRate * (accumBiases[l][j] / n);

        for (ulong k = 0; k < prevNumNeurons; k++) {
          this->parameters.weights[l][j][k] -= learningRate * (accumWeights[l][j][k] / n);
        }
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples,
                                T sampleLoss, T epochLoss, QMutex& callbackMutex)
{
  if (!this->trainingCallback) {
    return;
  }

  QMutexLocker locker(&callbackMutex);

  TrainingProgress<T> progress;
  progress.currentEpoch = currentEpoch;
  progress.totalEpochs = totalEpochs;
  progress.currentSample = currentSample;
  progress.totalSamples = totalSamples;
  progress.sampleLoss = sampleLoss;
  progress.epochLoss = epochLoss;

  this->trainingCallback(progress);
}

//===================================================================================================================//
// Step-by-step training methods (for external orchestration, e.g., CNN)
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreCPU<T>::backpropagate(const Output<T>& output)
{
  return this->stepWorker->backpropagateAndReturnInputGradients(output);
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::accumulate()
{
  this->stepWorker->accumulate();
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::resetAccumulators()
{
  this->stepWorker->resetAccumulators();
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::CoreCPU<int>;
template class ANN::CoreCPU<double>;
template class ANN::CoreCPU<float>;
