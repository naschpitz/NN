#include "ANN_CoreCPU.hpp"
#include "Common/Common_TrainingMonitor.hpp"
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
using namespace Common;

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
  // NOT here. This ensures the step-by-step path sees empty global accumulators,
  // so update() correctly reads from stepWorker's accumulators.
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::runWorkers(int numThreads, const std::function<void(int)>& body)
{
  // Dispatch numThreads worker chunks across this core's own pool (not the global one),
  // so a validation test() called from inside train()'s callback runs on a separate core's
  // pool and never starves on worker threads held by the enclosing train().
  if (this->workerPool.maxThreadCount() < numThreads)
    this->workerPool.setMaxThreadCount(numThreads);

  QVector<int> workerIndices(numThreads);
  std::iota(workerIndices.begin(), workerIndices.end(), 0);

  QtConcurrent::blockingMap(&this->workerPool, workerIndices, [&body](int workerIdx) { body(workerIdx); });
}

//===================================================================================================================//

template <typename T>
PredictResults<T> CoreCPU<T>::predict(ulong numSamples, const InputProvider<T>& provider)
{
  this->predictStart();

  int numThreads = this->numThreads;

  if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  // Use stepWorker as worker[0] so its internal state stays current for the
  // step-by-step training path (predict → backpropagate → accumulate → update).
  // Create additional workers only if more threads are needed. Workers are
  // allocated once and reused across batches.
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> extraWorkers;

  for (int i = 1; i < numThreads; i++) {
    extraWorkers.push_back(std::make_unique<CoreCPUWorker<T>>(this->layersConfig, this->trainingConfig,
                                                              this->parameters, this->costFunctionConfig, false));
  }

  std::vector<CoreCPUWorker<T>*> workerPtrs;
  workerPtrs.push_back(this->stepWorker.get());

  for (auto& w : extraWorkers)
    workerPtrs.push_back(w.get());

  PredictResults<T> results;
  results.reserve(numSamples);

  // Match the test() pattern: pull batches from the provider and process each
  // with QtConcurrent across the worker pool. Bounded memory regardless of
  // total dataset size — only one batch lives in memory at a time.
  ulong batchSize = std::max<ulong>(static_cast<ulong>(numThreads), this->testConfig.batchSize);
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;
  std::atomic<ulong> completedInputs{0};

  for (ulong batchIndex = 0; batchIndex < numBatches; batchIndex++) {
    Inputs<T> batch = provider(batchSize, batchIndex);
    ulong batchN = batch.size();

    if (batchN == 0)
      break;

    PredictResults<T> batchResults(batchN);

    // Distribute this batch across workers in contiguous chunks (extras to first workers).
    std::vector<ulong> workerInputCounts(numThreads);

    for (int i = 0; i < numThreads; i++)
      workerInputCounts[i] = batchN / static_cast<ulong>(numThreads) +
                             (static_cast<ulong>(i) < batchN % static_cast<ulong>(numThreads) ? 1 : 0);

    this->runWorkers(numThreads, [&](int workerIdx) {
      CoreCPUWorker<T>& worker = *workerPtrs[workerIdx];

      ulong workerLocalStart = 0;

      for (int i = 0; i < workerIdx; i++)
        workerLocalStart += workerInputCounts[i];
      ulong workerLocalEnd = workerLocalStart + workerInputCounts[workerIdx];

      for (ulong idx = workerLocalStart; idx < workerLocalEnd; idx++) {
        worker.propagate(batch[idx]);
        batchResults[idx].output = worker.getOutput();
        batchResults[idx].logits = worker.getOutputLogits();

        ulong completed = ++completedInputs;

        if (this->progressCallback)
          this->progressCallback(completed, numSamples);
      }
    });

    for (auto& r : batchResults)
      results.push_back(std::move(r));
  }

  this->predictEnd();

  return results;
}

//===================================================================================================================//

template <typename T>
PredictResult<T> CoreCPU<T>::predict(const Input<T>& input)
{
  // Direct single-input predict using stepWorker — avoids QtConcurrent::blockingMap
  // to prevent deadlocks when called from inside another blockingMap (e.g., CNN training).
  // We still record predictMetadata so getPredictMetadata() returns timing info on
  // every code path, matching the streaming overload's behaviour.
  this->predictStart();

  this->stepWorker->propagate(input);

  PredictResult<T> result;
  result.output = this->stepWorker->getOutput();
  result.logits = this->stepWorker->getOutputLogits();

  this->predictEnd();
  return result;
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

  // Mutex for serializing callback calls (prevents I/O contention)
  QMutex callbackMutex;

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  // Reproducible when trainingConfig.shuffleSeed != 0; non-deterministic otherwise.
  std::mt19937 rng(this->trainingConfig.shuffleSeed != 0 ? this->trainingConfig.shuffleSeed : std::random_device{}());

  // Create training monitor if monitoring is enabled
  const MonitoringConfig& monitoringConfig = this->trainingConfig.monitoringConfig;
  std::unique_ptr<TrainingMonitor<T>> monitor;

  if (monitoringConfig.enabled) {
    monitor = std::make_unique<TrainingMonitor<T>>(monitoringConfig);
  }

  for (ulong e = this->trainingConfig.startingEpoch; e < numEpochs && !this->stopRequested.load(); e++) {
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

      // Distribute this batch across workers in contiguous chunks (extras to first
      // workers). Each worker processes its chunk end-to-end on this core's own pool.
      // numThreads==1 means a single worker handles all samples in order — the
      // bit-deterministic path the tests rely on when they set shuffleSeed.
      std::vector<ulong> workerSampleCounts(numThreads);

      for (int i = 0; i < numThreads; i++)
        workerSampleCounts[i] = currentBatchSize / static_cast<ulong>(numThreads) +
                                (static_cast<ulong>(i) < currentBatchSize % static_cast<ulong>(numThreads) ? 1 : 0);

      this->runWorkers(numThreads, [&](int workerIdx) {
        CoreCPUWorker<T>& worker = *workers[workerIdx];

        ulong workerLocalStart = 0;

        for (int i = 0; i < workerIdx; i++)
          workerLocalStart += workerSampleCounts[i];
        ulong workerLocalEnd = workerLocalStart + workerSampleCounts[workerIdx];

        for (ulong localIdx = workerLocalStart; localIdx < workerLocalEnd; localIdx++) {
          const Sample<T>& sample = batchSamples[localIdx];
          worker.propagate(sample.input, true);
          T sampleLoss = worker.computeLoss(sample.output);
          worker.backpropagate(sample.output);
          worker.addToAccumLoss(sampleLoss);
          worker.accumulate();

          ulong completed = ++completedSamples;
          this->reportProgress(e + 1, numEpochs, completed, numSamples, sampleLoss, 0, callbackMutex);
        }
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

    // Store final loss from the last epoch
    this->trainingMetadata.finalLoss = avgEpochLoss;

    // Check training health if monitor is active
    bool shouldStop = false;

    if (monitor) {
      shouldStop = monitor->checkEpoch(e + 1, avgEpochLoss);

      // Build epoch progress with monitoring signals
      TrainingProgressEvent<T> progress;
      progress.currentEpoch = e + 1;
      progress.totalEpochs = numEpochs;
      progress.currentSample = numSamples;
      progress.totalSamples = numSamples;
      progress.sampleLoss = 0;
      progress.epochLoss = avgEpochLoss;
      progress.isNewBest = monitor->isNewBest();

      if (shouldStop) {
        progress.stoppedEarly = true;
        this->trainingMetadata.stopReason = monitor->getStopReason();
      }

      this->trainingMetadata.bestEpoch = monitor->getBestEpoch();
      this->trainingMetadata.bestLoss = monitor->getBestLoss();

      if (this->trainingCallback) {
        this->trainingCallback(progress);
      }
    } else {
      this->reportProgress(e + 1, numEpochs, numSamples, numSamples, 0, avgEpochLoss, callbackMutex);
    }

    // Always track the 0-based index of the last completed epoch (matches
    // EpochRecord::epoch), regardless of monitoring
    this->trainingMetadata.lastEpoch = e;

    // Record epoch history
    EpochRecord<T> epochRecord;
    epochRecord.epoch = e;
    epochRecord.loss = avgEpochLoss;
    epochRecord.valLoss = static_cast<T>(0);
    epochRecord.hasValLoss = false;
    epochRecord.isBest = monitor ? monitor->isNewBest() : false;
    epochRecord.completionTime =
      static_cast<ulong>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    this->trainingMetadata.epochHistory.push_back(epochRecord);

    // Notify the consumer that epoch e (0-based) is complete, so it can run
    // epoch-boundary work (validation, checkpoints) against the synced params.
    if (this->epochCompletedCallback) {
      EpochCompletionEvent<T> completion;
      completion.epoch = e;
      completion.totalEpochs = numEpochs;
      completion.epochLoss = avgEpochLoss;
      completion.isNewBest = monitor ? monitor->isNewBest() : false;
      completion.stoppedEarly = shouldStop;
      this->epochCompletedCallback(completion);
    }

    if (shouldStop) {
      break;
    }
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
    ulong batchN = batch.size();

    // Per-worker loss and correct counters
    std::vector<T> workerLosses(numThreads, 0);
    std::vector<ulong> workerCorrects(numThreads, 0);

    // Distribute this batch across workers in contiguous chunks (extras to first workers).
    std::vector<ulong> workerSampleCounts(numThreads);

    for (int i = 0; i < numThreads; i++)
      workerSampleCounts[i] = batchN / static_cast<ulong>(numThreads) +
                              (static_cast<ulong>(i) < batchN % static_cast<ulong>(numThreads) ? 1 : 0);

    this->runWorkers(numThreads, [&, numLayers](int workerIdx) {
      CoreCPUWorker<T>& worker = *workers[workerIdx];

      ulong workerLocalStart = 0;

      for (int i = 0; i < workerIdx; i++)
        workerLocalStart += workerSampleCounts[i];
      ulong workerLocalEnd = workerLocalStart + workerSampleCounts[workerIdx];

      for (ulong idx = workerLocalStart; idx < workerLocalEnd; idx++) {
        const Sample<T>& sample = batch[idx];
        worker.propagate(sample.input);
        T sampleLoss = worker.computeLoss(sample.output);

        workerLosses[workerIdx] += sampleLoss;

        const auto& outputActvs = worker.getActvs()[numLayers - 1];
        auto predIdx = std::distance(outputActvs.begin(), std::max_element(outputActvs.begin(), outputActvs.end()));
        auto expIdx =
          std::distance(sample.output.begin(), std::max_element(sample.output.begin(), sample.output.end()));

        if (predIdx == expIdx)
          workerCorrects[workerIdx]++;
      }
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

  // Deterministic seed for reproducible weight initialization
  std::mt19937 gen(42);

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
  // - step-by-step path accumulates directly into stepWorker's accumulators
  const Tensor3D<T>& accumWeights =
    this->accum_dCost_dWeights.empty() ? this->stepWorker->getAccumWeights() : this->accum_dCost_dWeights;
  const Tensor2D<T>& accumBiases =
    this->accum_dCost_dBiases.empty() ? this->stepWorker->getAccumBiases() : this->accum_dCost_dBiases;

  T n = static_cast<T>(numSamples);

  if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
    // Lazily allocate ADAM state on first update() call.
    // This handles the step-by-step path where train() is never called.
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

  TrainingProgressEvent<T> progress;
  progress.currentEpoch = currentEpoch;
  progress.totalEpochs = totalEpochs;
  progress.currentSample = currentSample;
  progress.totalSamples = totalSamples;
  progress.sampleLoss = sampleLoss;
  progress.epochLoss = epochLoss;

  this->trainingCallback(progress);
}

//===================================================================================================================//
//-- Step-by-step training (for external orchestration) --//
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreCPU<T>::backpropagate(const Output<T>& expected)
{
  return this->stepWorker->backpropagateAndReturnInputGradients(expected);
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
