#include "ANN_CoreGPUWorker.hpp"
#include "ANN_Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include <OCLW_Core.hpp>

using namespace ANN;

//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                                const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                                ulong progressReports, LogLevel logLevel)
  : layersConfig(layersConfig),
    trainingConfig(trainingConfig),
    parameters(parameters),
    costFunctionConfig(costFunctionConfig),
    progressReports(progressReports),
    logLevel(logLevel)
{
  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();
  this->core->setVerbose(this->logLevel >= LogLevel::DEBUG);

  this->bufferManager = std::make_unique<GPUBufferManager<T>>(
    this->core, this->layersConfig, this->parameters, this->trainingConfig, this->costFunctionConfig, this->logLevel);

  this->kernelBuilder =
    std::make_unique<GPUKernelBuilder<T>>(this->core, this->layersConfig, this->parameters, this->trainingConfig,
                                          this->costFunctionConfig, *this->bufferManager, this->logLevel);

  this->bufferManager->initializeParameters();
  this->bufferManager->loadSources(false);
  this->bufferManager->allocateBuffers();
}

//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                                const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                                OpenCLWrapper::Core& sharedCore, ulong progressReports, LogLevel logLevel)
  : layersConfig(layersConfig),
    trainingConfig(trainingConfig),
    parameters(parameters),
    costFunctionConfig(costFunctionConfig),
    progressReports(progressReports),
    logLevel(logLevel),
    core(&sharedCore)
{
  this->bufferManager = std::make_unique<GPUBufferManager<T>>(
    this->core, this->layersConfig, this->parameters, this->trainingConfig, this->costFunctionConfig, this->logLevel);

  this->kernelBuilder =
    std::make_unique<GPUKernelBuilder<T>>(this->core, this->layersConfig, this->parameters, this->trainingConfig,
                                          this->costFunctionConfig, *this->bufferManager, this->logLevel);

  // Shared-core mode: only initialize parameters.
  // Caller must invoke loadSources() and allocateBuffers() manually.
  this->bufferManager->initializeParameters();
}

//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input)
{
  // Set up predict kernels if not done yet
  if (!this->kernelBuilder->predictKernelsSetup) {
    this->kernelBuilder->setupPredictKernels();
    this->kernelBuilder->predictKernelsSetup = true;
  }

  // Write input to GPU and run forward pass
  this->core->template writeBuffer<T>("actvs", input, 0);

  // Execute predict kernels
  this->core->run();

  return this->bufferManager->readOutput();
}

//===================================================================================================================//
//-- Training (called by CoreGPU orchestrator) --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::trainSubset(const Samples<T>& batchSamples, ulong totalSamples, ulong epoch, ulong totalEpochs,
                                const TrainingCallback<T>& callback)
{
  ulong numSamplesInSubset = batchSamples.size();

  // Set up training kernels if not done yet
  if (!this->kernelBuilder->trainingKernelsSetup) {
    this->kernelBuilder->setupTrainingKernels();
    this->kernelBuilder->trainingKernelsSetup = true;
  }

  T subsetLoss = 0;

  // Reset accumulators at the start
  this->resetAccumulators();

  for (ulong s = 0; s < numSamplesInSubset; s++) {
    const Input<T>& input = batchSamples[s].input;
    const Output<T>& output = batchSamples[s].output;

    // Write input and expected output to GPU buffers
    this->core->template writeBuffer<T>("actvs", input, 0);
    this->core->template writeBuffer<T>("outputs", output, 0);

    // Generate and upload dropout mask (different mask per sample)
    if (this->bufferManager->hasDropout)
      this->bufferManager->generateAndUploadDropoutMask();

    // Execute all training kernels (forward pass + backward pass + gradient accumulation)
    this->core->run();

    // Calculate loss after kernels have run
    Output<T> predicted = this->bufferManager->readOutput();
    T sampleLoss = this->calculateLoss(predicted, output);
    subsetLoss += sampleLoss;

    // Report progress
    if (callback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = epoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = s + 1;
      progress.totalSamples = totalSamples;
      progress.sampleLoss = sampleLoss;
      progress.epochLoss = 0; // Not complete yet
      callback(progress);
    }
  }

  return subsetLoss;
}

//===================================================================================================================//
//-- Testing (called by CoreGPU orchestrator) --//
//===================================================================================================================//

template <typename T>
std::pair<T, ulong> CoreGPUWorker<T>::testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx)
{
  // Set up predict kernels if not done yet (forward pass only)
  if (!this->kernelBuilder->predictKernelsSetup) {
    this->kernelBuilder->setupPredictKernels();
    this->kernelBuilder->predictKernelsSetup = true;
  }

  T subsetLoss = 0;
  ulong subsetCorrect = 0;

  for (ulong s = startIdx; s < endIdx; s++) {
    const Input<T>& input = samples[s].input;
    const Output<T>& output = samples[s].output;

    // Write input to GPU buffer
    this->core->template writeBuffer<T>("actvs", input, 0);

    // Execute forward pass kernels only
    this->core->run();

    // Read predicted output for loss and accuracy computation
    Output<T> predicted = this->bufferManager->readOutput();
    T sampleLoss = this->calculateLoss(predicted, output);
    subsetLoss += sampleLoss;
    auto predIdx = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
    auto expIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    if (predIdx == expIdx)
      subsetCorrect++;
  }

  return {subsetLoss, subsetCorrect};
}

//===================================================================================================================//
//-- Batch predict (called by CoreGPU orchestrator) --//
//===================================================================================================================//

template <typename T>
Outputs<T> CoreGPUWorker<T>::predictSubset(const Inputs<T>& inputs, ulong startIdx, ulong endIdx,
                                           const ProgressCallback& callback)
{
  // Set up predict kernels if not done yet (forward pass only)
  if (!this->kernelBuilder->predictKernelsSetup) {
    this->kernelBuilder->setupPredictKernels();
    this->kernelBuilder->predictKernelsSetup = true;
  }

  Outputs<T> outputs;
  outputs.reserve(endIdx - startIdx);

  for (ulong i = startIdx; i < endIdx; i++) {
    const Input<T>& input = inputs[i];

    // Write input to GPU buffer
    this->core->template writeBuffer<T>("actvs", input, 0);

    // Execute forward pass kernels only
    this->core->run();

    // Read predicted output
    Output<T> predicted = this->bufferManager->readOutput();
    outputs.push_back(std::move(predicted));

    if (callback)
      callback(i - startIdx + 1, endIdx - startIdx);
  }

  return outputs;
}

//===================================================================================================================//
//-- Step-by-step training (for external orchestration) --//
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreGPUWorker<T>::backpropagate(const Output<T>& expected)
{
  // Set up backpropagate kernels if not done yet
  if (!this->kernelBuilder->backpropagateKernelsSetup) {
    this->kernelBuilder->setupBackpropagateKernels();
  }

  // Write expected output to GPU buffers
  this->core->template writeBuffer<T>("outputs", expected, 0);

  // Execute forward pass + backpropagation + input gradient kernels
  this->core->run();

  // Read and return input layer gradients
  return this->bufferManager->readInputGradients();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::accumulate()
{
  // Set up accumulate kernels if not done yet
  if (!this->kernelBuilder->accumulateKernelsSetup) {
    this->kernelBuilder->setupAccumulateKernels();
  }

  // Execute accumulation kernels
  this->core->run();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::resetAccumulators()
{
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  T zero = static_cast<T>(0);

  // Fill GPU accumulator buffers with zeros (no host-side vector allocation)
  this->core->template fillBuffer<T>("accum_dCost_dBiases", zero, numBiases);
  this->core->template fillBuffer<T>("accum_dCost_dWeights", zero, numWeights);
}

//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples)
{
  // Set up update kernels if not done yet (this clears sample kernels)
  if (!this->kernelBuilder->updateKernelsSetup) {
    this->kernelBuilder->setupUpdateKernels(numSamples);
    this->kernelBuilder->updateKernelsSetup = true;
  }

  // Run update kernels
  this->core->run();

  // After update, we need to re-setup kernels for next epoch/predict
  // Reset the flags so they get set up again when needed
  this->kernelBuilder->invalidateAllKernelFlags();
}

//===================================================================================================================//
// Explicit template instantiations.
template class ANN::CoreGPUWorker<int>;
template class ANN::CoreGPUWorker<double>;
template class ANN::CoreGPUWorker<float>;
