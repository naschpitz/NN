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
      logLevel(logLevel) {

  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();
  this->core->setVerbose(this->logLevel >= LogLevel::DEBUG);

  this->initializeParameters();
  this->loadSources(false);
  this->allocateBuffers();
}

//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                                 const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                                 OpenCLWrapper::Core& sharedCore,
                                 ulong progressReports, LogLevel logLevel)
    : layersConfig(layersConfig),
      trainingConfig(trainingConfig),
      parameters(parameters),
      costFunctionConfig(costFunctionConfig),
      progressReports(progressReports),
      logLevel(logLevel),
      core(&sharedCore) {

  // Shared-core mode: only initialize parameters.
  // Caller must invoke loadSources() and allocateBuffers() manually.
  this->initializeParameters();
}

//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input) {
  // Set up predict kernels if not done yet
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
    this->predictKernelsSetup = true;
  }

  // Write input to GPU and run forward pass
  this->core->template writeBuffer<T>("actvs", input, 0);

  // Execute predict kernels
  this->core->run();

  return this->readOutput();
}

//===================================================================================================================//
//-- Training (called by CoreGPU orchestrator) --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::trainSubset(const Samples<T>& samples, const std::vector<ulong>& indices,
                                ulong startIdx, ulong endIdx, ulong epoch, ulong totalEpochs,
                                const TrainingCallback<T>& callback) {
  ulong numSamplesInSubset = endIdx - startIdx;
  ulong totalSamples = samples.size();

  // Set up training kernels if not done yet
  if (!this->trainingKernelsSetup) {
    this->setupTrainingKernels();
    this->trainingKernelsSetup = true;
  }

  T subsetLoss = 0;

  // Reset accumulators at the start
  this->resetAccumulators();

  for (ulong s = startIdx; s < endIdx; s++) {
    const Input<T>& input = samples[indices[s]].input;
    const Output<T>& output = samples[indices[s]].output;

    // Write input and expected output to GPU buffers
    this->core->template writeBuffer<T>("actvs", input, 0);
    this->core->template writeBuffer<T>("outputs", output, 0);

    // Execute all training kernels (forward pass + backward pass + gradient accumulation)
    this->core->run();

    // Calculate loss after kernels have run
    T sampleLoss = this->calculateLoss(output);
    subsetLoss += sampleLoss;

    // Report progress
    if (callback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = epoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = s + 1;  // Global sample index
      progress.totalSamples = totalSamples;
      progress.sampleLoss = sampleLoss;
      progress.epochLoss = 0;  // Not complete yet
      callback(progress);
    }
  }

  return subsetLoss;
}

//===================================================================================================================//
//-- Testing (called by CoreGPU orchestrator) --//
//===================================================================================================================//

template <typename T>
std::pair<T, ulong> CoreGPUWorker<T>::testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx) {
  // Set up predict kernels if not done yet (forward pass only)
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
    this->predictKernelsSetup = true;
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

    // Calculate loss after forward pass
    T sampleLoss = this->calculateLoss(output);
    subsetLoss += sampleLoss;

    // Read predicted output for accuracy computation
    Output<T> predicted = this->readOutput();
    auto predIdx = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
    auto expIdx = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    if (predIdx == expIdx)
      subsetCorrect++;
  }

  return {subsetLoss, subsetCorrect};
}

//===================================================================================================================//
//-- Step-by-step training methods (for external orchestration, e.g., CNN) --//
//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreGPUWorker<T>::backpropagate(const Output<T>& output) {
  // Set up backpropagate kernels if not done yet
  if (!this->backpropagateKernelsSetup) {
    this->setupBackpropagateKernels();
  }

  // Write input and expected output to GPU buffers
  this->core->template writeBuffer<T>("outputs", output, 0);

  // Execute forward pass + backpropagation + input gradient kernels
  this->core->run();

  // Read and return input layer gradients
  return this->readInputGradients();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::accumulate() {
  // Set up accumulate kernels if not done yet
  if (!this->accumulateKernelsSetup) {
    this->setupAccumulateKernels();
  }

  // Execute accumulation kernels
  this->core->run();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::resetAccumulators() {
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  // Create zero-filled vectors
  std::vector<T> zeroBiases(numBiases, static_cast<T>(0));
  std::vector<T> zeroWeights(numWeights, static_cast<T>(0));

  // Write zeros to GPU accumulator buffers
  this->core->template writeBuffer<T>("accum_dCost_dBiases", zeroBiases, 0);
  this->core->template writeBuffer<T>("accum_dCost_dWeights", zeroWeights, 0);
}

//===================================================================================================================//
//-- Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases) {
  ulong numWeights = Utils<T>::count(this->parameters.weights);
  ulong numBiases = Utils<T>::count(this->parameters.biases);

  accumWeights.resize(numWeights);
  accumBiases.resize(numBiases);

  this->core->template readBuffer<T>("accum_dCost_dWeights", accumWeights, 0);
  this->core->template readBuffer<T>("accum_dCost_dBiases", accumBiases, 0);
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases) {
  // Write gradients directly to GPU (replacing existing values)
  this->core->template writeBuffer<T>("accum_dCost_dWeights", accumWeights, 0);
  this->core->template writeBuffer<T>("accum_dCost_dBiases", accumBiases, 0);
}

//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples) {
  // Set up update kernels if not done yet (this clears sample kernels)
  if (!this->updateKernelsSetup) {
    this->setupUpdateKernels(numSamples);
    this->updateKernelsSetup = true;
  }

  // Run update kernels
  this->core->run();

  // After update, we need to re-setup kernels for next epoch/predict
  // Reset the flags so they get set up again when needed
  this->invalidateAllKernelFlags();
}

//===================================================================================================================//
//-- Parameter synchronization --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::syncParametersFromGPU() {
  // Read updated weights and biases from GPU back to CPU parameters
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  // Create flat vectors to read GPU data
  std::vector<T> flatBiases(numBiases);
  std::vector<T> flatWeights(numWeights);

  // Read from GPU buffers
  this->core->template readBuffer<T>("biases", flatBiases, 0);
  this->core->template readBuffer<T>("weights", flatWeights, 0);

  // Unflatten back to nested tensor structure
  Utils<T>::unflatten(flatBiases, this->parameters.biases);
  Utils<T>::unflatten(flatWeights, this->parameters.weights);
}

//===================================================================================================================//
//-- Initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::initializeParameters() {
  ulong numLayers = this->layersConfig.size();

  // Check if parameters were loaded from file (non-empty)
  bool hasLoadedParameters = !this->parameters.weights.empty() &&
                             this->parameters.weights.size() == numLayers;

  this->parameters.weights.resize(numLayers);
  this->parameters.biases.resize(numLayers);

  // Random number generator for weight initialization
  std::random_device rd;
  std::mt19937 gen(rd());

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // Get previous layer's neuron count from layersConfig
    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    // He initialization for ReLU, Xavier for sigmoid/tanh
    ActvFuncType actvFuncType = layer.actvFuncType;

    double stddev;
    if (actvFuncType == ActvFuncType::RELU) {
      stddev = std::sqrt(2.0 / static_cast<double>(prevNumNeurons));
    } else {
      stddev = std::sqrt(1.0 / static_cast<double>(prevNumNeurons));
    }

    std::normal_distribution<double> dist(0.0, stddev);

    this->parameters.weights[l].resize(numNeurons);
    this->parameters.biases[l].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; j++) {
      this->parameters.weights[l][j].resize(prevNumNeurons);

      // Initialize weights randomly if not loaded from file
      if (!hasLoadedParameters) {
        for (ulong k = 0; k < prevNumNeurons; k++) {
          this->parameters.weights[l][j][k] = static_cast<T>(dist(gen));
        }
        // Initialize biases to zero
        this->parameters.biases[l][j] = static_cast<T>(0);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::loadSources(bool skipDefines) {
  if (this->logLevel >= LogLevel::INFO) std::cout << "Loading OpenCL kernels...\n";

  // Resolve .cl file paths relative to the source file's directory (via __FILE__),
  // so the kernels are found regardless of the current working directory.
  std::string srcFile = __FILE__;
  std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);

  // Load source files in order - they will be concatenated by OpenCL.
  // When used with a shared core (e.g. CNN), skipDefines avoids redefining TYPE, ActvFuncType, Layer.
  if (!skipDefines) {
    this->core->addSourceFile(srcDir + "opencl/Defines.hpp.cl");
  }

  this->core->addSourceFile(srcDir + "opencl/ActvFunc.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/IdxHelper.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/Kernels.cpp.cl");

  if (this->logLevel >= LogLevel::INFO) std::cout << "OpenCL kernels loaded.\n";
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::allocateBuffers() {
  ulong numLayers = this->layersConfig.size();
  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();
  ulong totalNumWeights = Utils<T>::count(this->parameters.weights);
  ulong totalNumBiases = Utils<T>::count(this->parameters.biases);

  // Common buffers
  if (this->logLevel >= LogLevel::INFO) std::cout << "Allocating ANN buffers...";
  this->core->template allocateBuffer<T>("actvs", totalNumNeurons);
  this->core->template allocateBuffer<T>("weights", totalNumWeights);
  this->core->template allocateBuffer<T>("biases", totalNumBiases);
  this->core->template allocateBuffer<T>("zs", totalNumNeurons);
  this->core->template allocateBuffer<T>("dCost_dActvs", totalNumNeurons);

  // Layers configuration buffer
  // Each Layer is: ulong numNeurons (8 bytes) + ActvFuncType (4 bytes) + padding (4 bytes) = 16 bytes
  this->core->template allocateBuffer<Layer>("layers", numLayers);

  std::vector<Layer> layersVec(this->layersConfig.begin(), this->layersConfig.end());
  this->core->template writeBuffer<Layer>("layers", layersVec, 0);

  // Training buffers
  this->core->template allocateBuffer<T>("dCost_dWeights", totalNumWeights);
  this->core->template allocateBuffer<T>("accum_dCost_dWeights", totalNumWeights);
  this->core->template allocateBuffer<T>("dCost_dBiases", totalNumBiases);
  this->core->template allocateBuffer<T>("accum_dCost_dBiases", totalNumBiases);
  this->core->template allocateBuffer<T>("outputs", this->layersConfig[numLayers - 1].numNeurons);

  // Loss weights buffer (one weight per output neuron).
  // Always allocated: for squaredDifference, filled with 1.0 so the kernel needs no branching.
  ulong numOutputNeurons = this->layersConfig[numLayers - 1].numNeurons;
  this->core->template allocateBuffer<T>("lossWeights", numOutputNeurons);

  std::vector<T> lossWeightsVec(numOutputNeurons, static_cast<T>(1));
  if (!this->costFunctionConfig.weights.empty()) {
    lossWeightsVec = this->costFunctionConfig.weights;
  }
  this->core->template writeBuffer<T>("lossWeights", lossWeightsVec, 0);

  if (this->logLevel >= LogLevel::INFO) std::cout << "ANN buffers allocation done.\n";

  // Write initialized weights and biases to GPU buffers
  std::vector<T> flatWeights = Utils<T>::flatten(this->parameters.weights);
  std::vector<T> flatBiases = Utils<T>::flatten(this->parameters.biases);
  this->core->template writeBuffer<T>("weights", flatWeights, 0);
  this->core->template writeBuffer<T>("biases", flatBiases, 0);
}

//===================================================================================================================//
//-- Kernel setup --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupPredictKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupTrainingKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addBackpropagateKernels(false);
  this->addAccumulateKernels();

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupBackpropagateKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addBackpropagateKernels(true);

  this->backpropagateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupAccumulateKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addAccumulateKernels();

  this->accumulateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupUpdateKernels(ulong numSamples) {
  // Clear sample kernels and set up update kernels
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addUpdateKernels(numSamples);

  this->updateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addUpdateKernels(ulong numSamples) {
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  this->core->addKernel("update_biases", numBiases, 0);
  this->core->template addArgument<T>("update_biases", "biases");
  this->core->template addArgument<T>("update_biases", "accum_dCost_dBiases");
  this->core->template addArgument<ulong>("update_biases", numSamples);
  this->core->template addArgument<float>("update_biases", this->trainingConfig.learningRate);
  this->core->template addArgument<ulong>("update_biases", numBiases);

  this->core->addKernel("update_weights", numWeights, 0);
  this->core->template addArgument<T>("update_weights", "weights");
  this->core->template addArgument<T>("update_weights", "accum_dCost_dWeights");
  this->core->template addArgument<ulong>("update_weights", numSamples);
  this->core->template addArgument<float>("update_weights", this->trainingConfig.learningRate);
  this->core->template addArgument<ulong>("update_weights", numWeights);
}

//===================================================================================================================//
//-- Kernel building blocks --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addPropagateKernels() {
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    std::string calculate_zs_id = "calculate_zs_layer" + std::to_string(l);
    std::string calculate_actvs_id = "calculate_actvs_layer" + std::to_string(l);

    // calculate_zs kernel: computes weighted sum + bias for each neuron
    this->core->addKernel(calculate_zs_id, "calculate_zs", numNeurons, 0);
    this->core->template addArgument<T>(calculate_zs_id, "zs");
    this->core->template addArgument<T>(calculate_zs_id, "weights");
    this->core->template addArgument<T>(calculate_zs_id, "actvs");
    this->core->template addArgument<T>(calculate_zs_id, "biases");
    this->core->template addArgument<ulong>(calculate_zs_id, l);
    this->core->template addArgument<Layer>(calculate_zs_id, "layers");
    this->core->template addArgument<ulong>(calculate_zs_id, numLayers);

    // calculate_actvs kernel: applies activation function
    // Softmax requires a single work-item (layer-wide), element-wise uses one per neuron
    ulong actvWorkItems = (layer.actvFuncType == ActvFuncType::SOFTMAX) ? 1 : numNeurons;
    this->core->addKernel(calculate_actvs_id, "calculate_actvs", actvWorkItems, 0);

    this->core->template addArgument<T>(calculate_actvs_id, "actvs");
    this->core->template addArgument<T>(calculate_actvs_id, "zs");
    this->core->template addArgument<ulong>(calculate_actvs_id, l);
    this->core->template addArgument<Layer>(calculate_actvs_id, "layers");
    this->core->template addArgument<ulong>(calculate_actvs_id, numLayers);
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addBackpropagateKernels(bool includeInputGradients) {
  ulong numLayers = this->layersConfig.size();

  // Last layer kernels
  ulong l = numLayers - 1;
  const Layer& lastLayer = this->layersConfig[l];

  ulong numNeurons = lastLayer.numNeurons;
  ulong numBiases = numNeurons;
  ulong numWeights = Utils<T>::count(this->parameters.weights[l]);

  // calculate_dCost_dActv_last_layer
  this->core->addKernel("calculate_dCost_dActv_last_layer", numNeurons, 0);
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "dCost_dActvs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "actvs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "outputs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "lossWeights");
  this->core->template addArgument<ulong>("calculate_dCost_dActv_last_layer", this->layersConfig[numLayers - 1].numNeurons);
  this->core->template addArgument<Layer>("calculate_dCost_dActv_last_layer", "layers");
  this->core->template addArgument<ulong>("calculate_dCost_dActv_last_layer", numLayers);

  std::string dCost_dBias_last_id = "calculate_dCost_dBias_layer" + std::to_string(l);
  std::string dCost_dWeight_last_id = "calculate_dCost_dWeight_layer" + std::to_string(l);

  // calculate_dCost_dBias for last layer
  this->core->addKernel(dCost_dBias_last_id, "calculate_dCost_dBias", numBiases, 0);
  this->core->template addArgument<T>(dCost_dBias_last_id, "dCost_dBiases");
  this->core->template addArgument<T>(dCost_dBias_last_id, "actvs");
  this->core->template addArgument<T>(dCost_dBias_last_id, "zs");
  this->core->template addArgument<T>(dCost_dBias_last_id, "dCost_dActvs");
  this->core->template addArgument<ulong>(dCost_dBias_last_id, l);
  this->core->template addArgument<Layer>(dCost_dBias_last_id, "layers");
  this->core->template addArgument<ulong>(dCost_dBias_last_id, numLayers);

  // calculate_dCost_dWeight for last layer
  this->core->addKernel(dCost_dWeight_last_id, "calculate_dCost_dWeight", numWeights, 0);
  this->core->template addArgument<T>(dCost_dWeight_last_id, "dCost_dWeights");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "actvs");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "zs");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "dCost_dActvs");
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, l);
  this->core->template addArgument<Layer>(dCost_dWeight_last_id, "layers");
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, numLayers);

  // Hidden layers (from second-to-last to first hidden layer)
  for (ulong layer_idx = numLayers - 2; layer_idx >= 1; layer_idx--) {
    const Layer& curr_layer = this->layersConfig[layer_idx];

    ulong curr_numNeurons = curr_layer.numNeurons;
    ulong curr_numBiases = curr_numNeurons;
    ulong curr_numWeights = Utils<T>::count(this->parameters.weights[layer_idx]);

    std::string dCost_dActv_id = "calculate_dCost_dActv_layer" + std::to_string(layer_idx);
    std::string dCost_dBias_id = "calculate_dCost_dBias_layer" + std::to_string(layer_idx);
    std::string dCost_dWeight_id = "calculate_dCost_dWeight_layer" + std::to_string(layer_idx);

    // calculate_dCost_dActv
    this->core->addKernel(dCost_dActv_id, "calculate_dCost_dActv", curr_numNeurons, 0);
    this->core->template addArgument<T>(dCost_dActv_id, "dCost_dActvs");
    this->core->template addArgument<T>(dCost_dActv_id, "actvs");
    this->core->template addArgument<T>(dCost_dActv_id, "weights");
    this->core->template addArgument<T>(dCost_dActv_id, "zs");
    this->core->template addArgument<ulong>(dCost_dActv_id, layer_idx);
    this->core->template addArgument<Layer>(dCost_dActv_id, "layers");
    this->core->template addArgument<ulong>(dCost_dActv_id, numLayers);

    // calculate_dCost_dBias
    this->core->addKernel(dCost_dBias_id, "calculate_dCost_dBias", curr_numBiases, 0);
    this->core->template addArgument<T>(dCost_dBias_id, "dCost_dBiases");
    this->core->template addArgument<T>(dCost_dBias_id, "actvs");
    this->core->template addArgument<T>(dCost_dBias_id, "zs");
    this->core->template addArgument<T>(dCost_dBias_id, "dCost_dActvs");
    this->core->template addArgument<ulong>(dCost_dBias_id, layer_idx);
    this->core->template addArgument<Layer>(dCost_dBias_id, "layers");
    this->core->template addArgument<ulong>(dCost_dBias_id, numLayers);

    // calculate_dCost_dWeight
    this->core->addKernel(dCost_dWeight_id, "calculate_dCost_dWeight", curr_numWeights, 0);
    this->core->template addArgument<T>(dCost_dWeight_id, "dCost_dWeights");
    this->core->template addArgument<T>(dCost_dWeight_id, "actvs");
    this->core->template addArgument<T>(dCost_dWeight_id, "zs");
    this->core->template addArgument<T>(dCost_dWeight_id, "dCost_dActvs");
    this->core->template addArgument<ulong>(dCost_dWeight_id, layer_idx);
    this->core->template addArgument<Layer>(dCost_dWeight_id, "layers");
    this->core->template addArgument<ulong>(dCost_dWeight_id, numLayers);

    // Break condition for ulong loop (can't go negative)
    if (layer_idx == 1) break;
  }

  // Optionally compute input layer gradients (layer 0) for external orchestrators (e.g., CNN)
  if (includeInputGradients) {
    ulong inputNumNeurons = this->layersConfig[0].numNeurons;

    this->core->addKernel("calculate_dCost_dActv_layer0", "calculate_dCost_dActv", inputNumNeurons, 0);
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "dCost_dActvs");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "actvs");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "weights");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "zs");
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", static_cast<ulong>(0));
    this->core->template addArgument<Layer>("calculate_dCost_dActv_layer0", "layers");
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", numLayers);
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addAccumulateKernels() {
  ulong totalNumBiases = Utils<T>::count(this->parameters.biases);
  ulong totalNumWeights = Utils<T>::count(this->parameters.weights);

  this->core->addKernel("accumulate_dCost_dBiases", totalNumBiases, 0);
  this->core->template addArgument<T>("accumulate_dCost_dBiases", "accum_dCost_dBiases");
  this->core->template addArgument<T>("accumulate_dCost_dBiases", "dCost_dBiases");
  this->core->template addArgument<ulong>("accumulate_dCost_dBiases", totalNumBiases);

  this->core->addKernel("accumulate_dCost_dWeights", totalNumWeights, 0);
  this->core->template addArgument<T>("accumulate_dCost_dWeights", "accum_dCost_dWeights");
  this->core->template addArgument<T>("accumulate_dCost_dWeights", "dCost_dWeights");
  this->core->template addArgument<ulong>("accumulate_dCost_dWeights", totalNumWeights);
}

//===================================================================================================================//
//-- Helpers --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::invalidateAllKernelFlags() {
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->backpropagateKernelsSetup = false;
  this->accumulateKernelsSetup = false;
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
//-- Loss and output --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::calculateLoss(const Output<T>& expected) {
  // Read output activations from GPU
  Output<T> actual = this->readOutput();

  T loss = 0;

  for (ulong i = 0; i < expected.size(); i++) {
    T diff = actual[i] - expected[i];
    T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
    loss += weight * diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreGPUWorker<T>::readInputGradients() {
  // Read dCost_dActvs for the input layer (layer 0) from GPU
  // The dCost_dActvs buffer is laid out with all neurons contiguously,
  // and layer 0 starts at offset 0
  ulong inputNumNeurons = this->layersConfig[0].numNeurons;

  Tensor1D<T> inputGradients;
  inputGradients.resize(inputNumNeurons);

  this->core->template readBuffer<T>("dCost_dActvs", inputGradients, 0);

  return inputGradients;
}

//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::readOutput() {
  ulong numLayers = this->layersConfig.size();

  ulong outputOffset = 0;

  for (ulong l = 0; l < numLayers - 1; l++) {
    outputOffset += this->layersConfig[l].numNeurons;
  }

  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();
  ulong outputNumNeurons = totalNumNeurons - outputOffset;

  Output<T> output;
  output.resize(outputNumNeurons);

  this->core->readBuffer("actvs", output, outputOffset);

  return output;
}

//===================================================================================================================//
// Explicit template instantiations.
template class ANN::CoreGPUWorker<int>;
template class ANN::CoreGPUWorker<double>;
template class ANN::CoreGPUWorker<float>;

