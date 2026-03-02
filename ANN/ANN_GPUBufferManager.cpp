#include "ANN_GPUBufferManager.hpp"
#include "ANN_Utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace ANN;

//===================================================================================================================//

template <typename T>
GPUBufferManager<T>::GPUBufferManager(OpenCLWrapper::Core* core, const LayersConfig& layersConfig,
                                      Parameters<T>& parameters, const TrainingConfig<T>& trainingConfig,
                                      const CostFunctionConfig<T>& costFunctionConfig, LogLevel logLevel)
  : core(core),
    layersConfig(layersConfig),
    parameters(parameters),
    trainingConfig(trainingConfig),
    costFunctionConfig(costFunctionConfig),
    logLevel(logLevel)
{
}

//===================================================================================================================//
//-- Initialization --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::initializeParameters()
{
  ulong numLayers = this->layersConfig.size();

  // Check if parameters were loaded from file (non-empty)
  bool hasLoadedParameters = !this->parameters.weights.empty() && this->parameters.weights.size() == numLayers;

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
void GPUBufferManager<T>::loadSources(bool skipDefines)
{
  if (this->logLevel >= LogLevel::INFO)
    std::cout << "Loading OpenCL kernels...\n";

  // Resolve .cl file paths relative to the source file's directory (via __FILE__),
  // so the kernels are found regardless of the current working directory.
  std::string srcFile = __FILE__;
  std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);

  // Load source files in order - they will be concatenated by OpenCL.
  // When used with a shared core (e.g. CNN), skipDefines avoids redefining TYPE, ActvFuncType, Layer.
  if (!skipDefines) {
    this->core->addSourceFile(srcDir + "opencl/ANN_Defines.hpp.cl");
  }

  this->core->addSourceFile(srcDir + "opencl/ANN_IdxHelper.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/ANN_ActvFunc.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/ANN_Propagate.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/ANN_Backpropagate.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/ANN_Update.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/ANN_Loss.cpp.cl");

  if (this->logLevel >= LogLevel::INFO)
    std::cout << "OpenCL kernels loaded.\n";
}

//===================================================================================================================//

// Buffer allocation lives in the worker/buffer manager (not CoreGPU) because each worker owns
// its own OpenCLWrapper::Core bound to a specific GPU device. GPU buffers are device-local, so
// they must be allocated on the device that will use them. This differs from the CPU path,
// where CoreCPU allocates shared host memory that all CPU workers can reference directly.
template <typename T>
void GPUBufferManager<T>::allocateBuffers()
{
  ulong numLayers = this->layersConfig.size();
  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();
  ulong totalNumWeights = Utils<T>::count(this->parameters.weights);
  ulong totalNumBiases = Utils<T>::count(this->parameters.biases);

  // Common buffers
  if (this->logLevel >= LogLevel::INFO)
    std::cout << "Allocating ANN buffers...";
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

  // Dropout mask buffer (allocated if dropout is enabled)
  this->hasDropout = (this->trainingConfig.dropoutRate > 0.0f);

  if (this->hasDropout) {
    this->core->template allocateBuffer<T>("dropoutMask", totalNumNeurons);
  }

  // Adam optimizer buffers
  if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
    T zero = static_cast<T>(0);

    this->core->template allocateBuffer<T>("adam_m_weights", totalNumWeights);
    this->core->template allocateBuffer<T>("adam_v_weights", totalNumWeights);
    this->core->template allocateBuffer<T>("adam_m_biases", totalNumBiases);
    this->core->template allocateBuffer<T>("adam_v_biases", totalNumBiases);

    this->core->template fillBuffer<T>("adam_m_weights", zero, totalNumWeights);
    this->core->template fillBuffer<T>("adam_v_weights", zero, totalNumWeights);
    this->core->template fillBuffer<T>("adam_m_biases", zero, totalNumBiases);
    this->core->template fillBuffer<T>("adam_v_biases", zero, totalNumBiases);
  }

  if (this->logLevel >= LogLevel::INFO)
    std::cout << "ANN buffers allocation done.\n";

  // Write initialized weights and biases to GPU buffers
  std::vector<T> flatWeights = Utils<T>::flatten(this->parameters.weights);
  std::vector<T> flatBiases = Utils<T>::flatten(this->parameters.biases);
  this->core->template writeBuffer<T>("weights", flatWeights, 0);
  this->core->template writeBuffer<T>("biases", flatBiases, 0);
}

//===================================================================================================================//
//-- Parameter synchronization --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::syncParametersFromGPU()
{
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
//-- Data I/O --//
//===================================================================================================================//

template <typename T>
Output<T> GPUBufferManager<T>::readOutput()
{
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

template <typename T>
Tensor1D<T> GPUBufferManager<T>::readInputGradients()
{
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
//-- Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::readAccumulatedGradients(Tensor1D<T>& accumWeights, Tensor1D<T>& accumBiases)
{
  ulong numWeights = Utils<T>::count(this->parameters.weights);
  ulong numBiases = Utils<T>::count(this->parameters.biases);

  accumWeights.resize(numWeights);
  accumBiases.resize(numBiases);

  this->core->template readBuffer<T>("accum_dCost_dWeights", accumWeights, 0);
  this->core->template readBuffer<T>("accum_dCost_dBiases", accumBiases, 0);
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::setAccumulators(const Tensor1D<T>& accumWeights, const Tensor1D<T>& accumBiases)
{
  // Write gradients directly to GPU (replacing existing values)
  this->core->template writeBuffer<T>("accum_dCost_dWeights", accumWeights, 0);
  this->core->template writeBuffer<T>("accum_dCost_dBiases", accumBiases, 0);
}

//===================================================================================================================//
//-- Dropout mask generation and upload --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::generateAndUploadDropoutMask()
{
  ulong numLayers = this->layersConfig.size();
  float rate = this->trainingConfig.dropoutRate;

  T scale = static_cast<T>(1) / (static_cast<T>(1) - static_cast<T>(rate));
  std::bernoulli_distribution dist(1.0 - static_cast<double>(rate));

  // Build flat mask matching the flat actvs buffer layout
  ulong totalNeurons = 0;
  for (ulong l = 0; l < numLayers; l++)
    totalNeurons += this->layersConfig[l].numNeurons;

  std::vector<T> mask(totalNeurons);
  ulong offset = 0;

  for (ulong l = 0; l < numLayers; l++) {
    ulong numNeurons = this->layersConfig[l].numNeurons;

    // Apply dropout only to hidden layers (skip input layer 0 and output layer N-1)
    bool applyDropout = (l > 0 && l < numLayers - 1);

    for (ulong j = 0; j < numNeurons; j++) {
      mask[offset + j] = applyDropout ? (dist(this->dropoutRng) ? scale : static_cast<T>(0)) : static_cast<T>(1);
    }

    offset += numNeurons;
  }

  this->core->template writeBuffer<T>("dropoutMask", mask, 0);
}

//===================================================================================================================//
//-- Offset helpers --//
//===================================================================================================================//

template <typename T>
ulong GPUBufferManager<T>::getActvOffset(ulong layerIdx) const
{
  ulong offset = 0;
  for (ulong l = 0; l < layerIdx; l++)
    offset += this->layersConfig[l].numNeurons;
  return offset;
}

template <typename T>
ulong GPUBufferManager<T>::getWeightOffset(ulong layerIdx) const
{
  ulong offset = 0;
  for (ulong l = 1; l < layerIdx; l++)
    offset += this->layersConfig[l].numNeurons * this->layersConfig[l - 1].numNeurons;
  return offset;
}

template <typename T>
ulong GPUBufferManager<T>::getBiasOffset(ulong layerIdx) const
{
  ulong offset = 0;
  for (ulong l = 1; l < layerIdx; l++)
    offset += this->layersConfig[l].numNeurons;
  return offset;
}

template <typename T>
ulong GPUBufferManager<T>::getOutputActvOffset() const
{
  return this->getActvOffset(this->layersConfig.size() - 1);
}

template <typename T>
ulong GPUBufferManager<T>::getNumOutputNeurons() const
{
  return this->layersConfig.back().numNeurons;
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class ANN::GPUBufferManager<int>;
template class ANN::GPUBufferManager<float>;
template class ANN::GPUBufferManager<double>;