#include "ANN_CoreCPU.hpp"
#include "ANN_Utils.hpp"

#include <OCLW_Core.hpp>
#include <QFile>
#include <QThread>
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
CoreCPU<T>::CoreCPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig) {
  this->allocateCommon();

  switch (this->modeType) {
    case ModeType::TRAIN:
      this->allocateTraining();
      break;
    case ModeType::PREDICT:
    case ModeType::TEST:
      break;
  }
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::predict(const Input<T>& input) {
  this->predictStart();

  this->propagate(input);
  Output<T> output = this->getOutput();

  this->predictEnd();

  return output;
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(const Samples<T>& samples) {
  this->trainingStart(samples.size());

  ulong numSamples = samples.size();
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
  std::vector<SampleWorker<T>> workers(numThreads);

  for (int i = 0; i < numThreads; i++) {
    this->allocateWorker(workers[i]);
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
    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      // Reset global accumulators at the start of each batch
      this->resetAccumulators();

      // Reset worker accumulators at the start of each batch (including loss)
      for (int i = 0; i < numThreads; i++) {
        this->resetWorkerAccumulators(workers[i]);
        workers[i].accum_loss = 0;
      }

      // Build batch indices
      QVector<ulong> batchIndices(currentBatchSize);

      for (ulong s = 0; s < currentBatchSize; s++) {
        batchIndices[s] = sampleIndices[batchStart + s];
      }

      // Use blockingMap to process all samples in the batch in parallel
      QtConcurrent::blockingMap(batchIndices, [&, numThreads](ulong sampleIndex) {
        // Each thread gets a unique worker index on first use (thread_local persists for thread lifetime)
        thread_local int workerIndex = nextWorkerIndex.fetch_add(1) % numThreads;
        SampleWorker<T>& worker = workers[workerIndex];

        // Process the sample using the shared computation functions
        const Sample<T>& sample = samples[sampleIndex];
        this->propagate(sample.input, worker.actvs, worker.zs);
        worker.sampleLoss = this->calculateLoss(sample.output, worker.actvs);

        this->backpropagate(sample.output, worker.actvs, worker.zs,
                            worker.dCost_dActvs, worker.dCost_dWeights, worker.dCost_dBiases);

        // Accumulate loss to worker's local accumulator (no atomic needed - each thread has its own worker)
        worker.accum_loss += worker.sampleLoss;

        // Accumulate gradients to worker's local accumulators (no mutex needed)
        this->accumulateToWorker(worker);

        // Increment completed samples counter and report progress
        ulong completed = ++completedSamples;
        this->reportProgress(e + 1, numEpochs, completed, numSamples, worker.sampleLoss, 0, callbackMutex);
      });

      // Merge all worker accumulators into global accumulators
      for (int i = 0; i < numThreads; i++) {
        this->mergeWorkerAccumulators(workers[i]);
        epochLoss += workers[i].accum_loss;
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
TestResult<T> CoreCPU<T>::test(const Samples<T>& samples) {
  ulong numSamples = samples.size();

  // Use configured numThreads, or all available cores if 0
  int numThreads = this->numThreads;

  if (numThreads <= 0) {
    numThreads = QThreadPool::globalInstance()->maxThreadCount();
  }

  // Pre-allocate workers for each thread (only need actvs and zs for forward pass)
  struct TestWorker {
    Tensor2D<T> actvs;
    Tensor2D<T> zs;
    T accum_loss;
    ulong accum_correct;
  };

  std::vector<TestWorker> workers(numThreads);
  ulong numLayers = this->layersConfig.size();

  for (int i = 0; i < numThreads; i++) {
    workers[i].actvs.resize(numLayers);
    workers[i].zs.resize(numLayers);
    workers[i].accum_loss = 0;
    workers[i].accum_correct = 0;

    for (ulong l = 0; l < numLayers; l++) {
      Layer layer = this->layersConfig[l];
      ulong numNeurons = layer.numNeurons;
      workers[i].actvs[l].resize(numNeurons);
    }

    for (ulong l = 1; l < numLayers; l++) {
      Layer layer = this->layersConfig[l];
      ulong numNeurons = layer.numNeurons;
      workers[i].zs[l].resize(numNeurons);
    }
  }

  // Atomic counter for assigning unique worker indices to threads
  std::atomic<int> nextWorkerIndex{0};

  // Pre-allocate sample indices vector
  QVector<ulong> sampleIndices(numSamples);

  for (ulong s = 0; s < numSamples; s++) {
    sampleIndices[s] = s;
  }

  // Use blockingMap to process all samples in parallel
  QtConcurrent::blockingMap(sampleIndices, [&, numThreads](ulong sampleIndex) {
    // Each thread gets a unique worker index on first use
    thread_local int workerIndex = nextWorkerIndex.fetch_add(1) % numThreads;
    TestWorker& worker = workers[workerIndex];

    // Process the sample - forward pass only
    const Sample<T>& sample = samples[sampleIndex];
    this->propagate(sample.input, worker.actvs, worker.zs);
    T sampleLoss = this->calculateLoss(sample.output, worker.actvs);

    // Accumulate loss
    worker.accum_loss += sampleLoss;

    // Accuracy: compare argmax of predicted vs expected
    const auto& outputActvs = worker.actvs[numLayers - 1];
    auto predIdx = std::distance(outputActvs.begin(), std::max_element(outputActvs.begin(), outputActvs.end()));
    auto expIdx = std::distance(sample.output.begin(), std::max_element(sample.output.begin(), sample.output.end()));

    if (predIdx == expIdx)
      worker.accum_correct++;
  });

  // Sum up losses and correct counts from all workers
  T totalLoss = 0;
  ulong totalCorrect = 0;

  for (int i = 0; i < numThreads; i++) {
    totalLoss += workers[i].accum_loss;
    totalCorrect += workers[i].accum_correct;
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
// Functions used in init()
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateCommon() {
  ulong numLayers = this->layersConfig.size();

  // Check if parameters were loaded from file (non-empty)
  bool hasLoadedParameters = !this->parameters.weights.empty() &&
                             this->parameters.weights.size() == numLayers;

  this->actvs.resize(numLayers);
  this->zs.resize(numLayers);

  // Only resize parameters if not loaded from file
  if (!hasLoadedParameters) {
    this->parameters.weights.resize(numLayers);
    this->parameters.biases.resize(numLayers);
  }

  for (ulong l = 0; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    this->actvs[l].resize(numNeurons);
  }

  // Random number generator for weight initialization
  std::random_device rd;
  std::mt19937 gen(rd());

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;
    ulong prevNumNeurons = this->actvs[l - 1].size();

    this->zs[l].resize(numNeurons);

    // Only initialize weights if not loaded from file
    if (!hasLoadedParameters) {
      this->parameters.weights[l].resize(numNeurons);
      this->parameters.biases[l].resize(numNeurons, static_cast<T>(0));

      // He initialization for ReLU, Xavier for sigmoid/tanh
      // stddev = sqrt(2 / fan_in) for He, sqrt(1 / fan_in) for Xavier
      ActvFuncType actvFuncType = layer.actvFuncType;
      double stddev;
      
      if (actvFuncType == ActvFuncType::RELU) {
        stddev = std::sqrt(2.0 / static_cast<double>(prevNumNeurons));
      } else {
        stddev = std::sqrt(1.0 / static_cast<double>(prevNumNeurons));
      }

      // Use double for the distribution, then cast to T
      std::normal_distribution<double> dist(0.0, stddev);

      for (ulong j = 0; j < numNeurons; j++) {
        this->parameters.weights[l][j].resize(prevNumNeurons);

        // Initialize weights with random values
        for (ulong k = 0; k < prevNumNeurons; k++) {
          this->parameters.weights[l][j][k] = static_cast<T>(dist(gen));
        }
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateTraining() {
  ulong numLayers = this->layersConfig.size();

  this->dCost_dActvs.resize(numLayers);
  this->accum_dCost_dWeights.resize(numLayers);

  this->dCost_dWeights.resize(numLayers);
  this->dCost_dBiases.resize(numLayers);
  this->accum_dCost_dBiases.resize(numLayers);

  // Allocate dCost_dActvs for input layer (needed for backpropagate() to return input gradients)
  this->dCost_dActvs[0].resize(this->layersConfig[0].numNeurons);

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    this->dCost_dActvs[l].resize(numNeurons);
    this->dCost_dWeights[l].resize(numNeurons);
    this->accum_dCost_dWeights[l].resize(numNeurons);

    this->dCost_dBiases[l].resize(numNeurons);
    this->accum_dCost_dBiases[l].resize(numNeurons);

    // The number of neurons is the same as the number of activations.
    ulong prevNumNeurons = this->actvs[l - 1].size();

    for (ulong j = 0; j < numNeurons; j++) {
      this->dCost_dWeights[l][j].resize(prevNumNeurons);
      this->accum_dCost_dWeights[l][j].resize(prevNumNeurons);
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateWorker(SampleWorker<T>& worker) {
  ulong numLayers = this->layersConfig.size();

  worker.actvs.resize(numLayers);
  worker.zs.resize(numLayers);
  worker.dCost_dActvs.resize(numLayers);
  worker.dCost_dWeights.resize(numLayers);
  worker.dCost_dBiases.resize(numLayers);
  worker.accum_dCost_dWeights.resize(numLayers);
  worker.accum_dCost_dBiases.resize(numLayers);

  for (ulong l = 0; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    worker.actvs[l].resize(numNeurons);
  }

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;
    ulong prevNumNeurons = worker.actvs[l - 1].size();

    worker.zs[l].resize(numNeurons);
    worker.dCost_dActvs[l].resize(numNeurons);
    worker.dCost_dWeights[l].resize(numNeurons);
    worker.dCost_dBiases[l].resize(numNeurons);
    worker.accum_dCost_dWeights[l].resize(numNeurons);
    worker.accum_dCost_dBiases[l].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; j++) {
      worker.dCost_dWeights[l][j].resize(prevNumNeurons);
      worker.accum_dCost_dWeights[l][j].resize(prevNumNeurons);
    }
  }
}

//===================================================================================================================//
// Core computation functions (parameterized - used by both single-threaded and multi-threaded paths)
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::propagate(const Input<T>& input, Tensor2D<T>& actvs, Tensor2D<T>& zs) {
  ulong numLayers = this->layersConfig.size();

  // Set the actvs values of the Neurons of the first layer the same values as the input.
  actvs[0] = input;

  // Propagate from the second layer on, as the first layer is input only.
  for (ulong l = 1; l < numLayers; l++) {
    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // Compute weighted sums (zs) for all neurons in this layer
    for (ulong j = 0; j < numNeurons; j++) {
      zs[l][j] = this->parameters.biases[l][j];  // Start with bias

      for (ulong k = 0; k < prevNumNeurons; k++) {
        zs[l][j] += this->parameters.weights[l][j][k] * actvs[l - 1][k];
      }
    }

    // Apply activation function (handles all types including softmax)
    const T* zsData = zs[l].data();
    T* actvsData = actvs[l].data();
    ActvFuncType actvFuncType = layer.actvFuncType;

    ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, false);
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::backpropagate(const Output<T>& output, const Tensor2D<T>& actvs, const Tensor2D<T>& zs,
                                Tensor2D<T>& dCost_dActvs, Tensor3D<T>& dCost_dWeights, Tensor2D<T>& dCost_dBiases) {
  ulong numLayers = this->layersConfig.size();

  // For the last layer, calculate dCost_dActv
  ulong l = numLayers - 1;

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;

  for (ulong j = 0; j < numNeurons; j++) {
    dCost_dActvs[l][j] = this->calc_dCost_dActv(j, output, actvs);
  }

  for (ulong j = 0; j < numNeurons; j++) {
    dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j, actvs, zs, dCost_dActvs);

    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    for (ulong k = 0; k < prevNumNeurons; k++) {
      dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k, actvs, zs, dCost_dActvs);
    }
  }

  // For the remaining layers, calculate backwards
  for (ulong l = numLayers - 2; l >= 1; l--) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // First pass: compute all dCost_dActvs for this layer
    for (ulong j = 0; j < numNeurons; j++) {
      dCost_dActvs[l][j] = this->calc_dCost_dActv(l, j, actvs, zs, dCost_dActvs);
    }

    // Second pass: compute bias and weight gradients
    for (ulong j = 0; j < numNeurons; j++) {
      dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j, actvs, zs, dCost_dActvs);

      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      for (ulong k = 0; k < prevNumNeurons; k++) {
        dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k, actvs, zs, dCost_dActvs);
      }
    }

    if (l == 1) break;
  }
}

//===================================================================================================================//

// Particular case for the last layer.
template <typename T>
T CoreCPU<T>::calc_dCost_dActv(ulong j, const Output<T>& output, const Tensor2D<T>& actvs) {
  ulong numLayers = this->layersConfig.size();
  ulong l = numLayers - 1;

  T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[j] : static_cast<T>(1);
  return 2 * weight * (actvs[l][j] - output[j]);
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calc_dCost_dActv(ulong l, ulong k, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs) {
  const Layer& nextLayer = this->layersConfig[l + 1];
  ulong nextNumNeurons = nextLayer.numNeurons;
  ActvFuncType actvFuncType = nextLayer.actvFuncType;

  const T* zsData = zs[l + 1].data();
  T* actvsData = const_cast<T*>(actvs[l + 1].data());
  const T* dCost_dActvsData = dCost_dActvs[l + 1].data();

  Tensor1D<T> dCost_dZs(nextNumNeurons);
  T* dCost_dZsData = dCost_dZs.data();

  ActvFunc::calculate(zsData, actvsData, nextNumNeurons, actvFuncType, true, dCost_dActvsData, dCost_dZsData);

  T sum = 0;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    T weight = this->parameters.weights[l + 1][j][k];
    T dCost_dZ = dCost_dZs[j];

    sum += weight * dCost_dZ;
  }

  return sum;
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calc_dCost_dWeight(ulong l, ulong j, ulong k, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs) {
  T actv = actvs[l - 1][k];

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;
  ActvFuncType actvFuncType = layer.actvFuncType;

  const T* zsData = zs[l].data();
  T* actvsData = const_cast<T*>(actvs[l].data());
  const T* dCost_dActvsData = dCost_dActvs[l].data();

  Tensor1D<T> dCost_dZs(numNeurons);
  T* dCost_dZsData = dCost_dZs.data();

  ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, true, dCost_dActvsData, dCost_dZsData);

  T dCost_dZ = dCost_dZs[j];

  return actv * dCost_dZ;
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calc_dCost_dBias(ulong l, ulong j, const Tensor2D<T>& actvs, const Tensor2D<T>& zs, const Tensor2D<T>& dCost_dActvs) {
  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;
  ActvFuncType actvFuncType = layer.actvFuncType;

  const T* zsData = zs[l].data();
  T* actvsData = const_cast<T*>(actvs[l].data());
  const T* dCost_dActvsData = dCost_dActvs[l].data();

  Tensor1D<T> dCost_dZs(numNeurons);
  T* dCost_dZsData = dCost_dZs.data();

  ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, true, dCost_dActvsData, dCost_dZsData);

  T dCost_dZ = dCost_dZs[j];

  return dCost_dZ;
}

//===================================================================================================================//
// Functions used by train()
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::resetAccumulators() {
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
void CoreCPU<T>::resetWorkerAccumulators(SampleWorker<T>& worker) {
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      worker.accum_dCost_dBiases[l][j] = static_cast<T>(0);

      for (ulong k = 0; k < prevNumNeurons; k++) {
        worker.accum_dCost_dWeights[l][j][k] = static_cast<T>(0);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::accumulateToWorker(SampleWorker<T>& worker) {
  // No mutex needed - each worker accumulates to its own local accumulators
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      worker.accum_dCost_dBiases[l][j] += worker.dCost_dBiases[l][j];

      for (ulong k = 0; k < prevNumNeurons; k++) {
        worker.accum_dCost_dWeights[l][j][k] += worker.dCost_dWeights[l][j][k];
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::mergeWorkerAccumulators(const SampleWorker<T>& worker) {
  // Called once per worker per epoch - minimal mutex contention
  QMutexLocker locker(&this->accumulatorMutex);

  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      this->accum_dCost_dBiases[l][j] += worker.accum_dCost_dBiases[l][j];

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->accum_dCost_dWeights[l][j][k] += worker.accum_dCost_dWeights[l][j][k];
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calculateLoss(const Output<T>& expected, const Tensor2D<T>& actvs) {
  ulong numLayers = this->layersConfig.size();
  const auto& outputActvs = actvs[numLayers - 1];

  T loss = 0;
  for (ulong i = 0; i < outputActvs.size(); i++) {
    T diff = outputActvs[i] - expected[i];
    T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
    loss += weight * diff * diff;
  }

  return loss / static_cast<T>(outputActvs.size());
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::update(ulong numSamples) {
  T learningRate = this->trainingConfig.learningRate;
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      this->parameters.biases[l][j] -= learningRate * (this->accum_dCost_dBiases[l][j] / numSamples);

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->parameters.weights[l][j][k] -= learningRate * (this->accum_dCost_dWeights[l][j][k] / numSamples);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::reportProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples,
                                 T sampleLoss, T epochLoss, QMutex& callbackMutex) {
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
// Convenience wrappers using member data (for predict())
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::propagate(const Input<T>& input) {
  this->propagate(input, this->actvs, this->zs);
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::getOutput() {
  ulong numLayers = this->layersConfig.size();

  return this->actvs[numLayers - 1];
}

//===================================================================================================================//
// Step-by-step training methods (for external orchestration, e.g., CNN)
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreCPU<T>::backpropagate(const Output<T>& output) {
  // Uses stored actvs/zs from the last predict() call
  // Computes gradients into member dCost_dActvs, dCost_dWeights, dCost_dBiases
  this->backpropagate(output, this->actvs, this->zs,
                      this->dCost_dActvs, this->dCost_dWeights, this->dCost_dBiases);

  // Additionally compute dCost_dActvs for the input layer (layer 0)
  // This is needed by external orchestrators (e.g., CNN) to continue backpropagation
  ulong inputNumNeurons = this->layersConfig[0].numNeurons;

  for (ulong k = 0; k < inputNumNeurons; k++) {
    this->dCost_dActvs[0][k] = this->calc_dCost_dActv(0, k, this->actvs, this->zs, this->dCost_dActvs);
  }

  return this->dCost_dActvs[0];
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::accumulate() {
  // Accumulates computed gradients (from backpropagate) into the member accumulators
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      this->accum_dCost_dBiases[l][j] += this->dCost_dBiases[l][j];

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->accum_dCost_dWeights[l][j][k] += this->dCost_dWeights[l][j][k];
      }
    }
  }
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::CoreCPU<int>;
template class ANN::CoreCPU<double>;
template class ANN::CoreCPU<float>;
