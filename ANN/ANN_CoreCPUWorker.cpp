#include "ANN_CoreCPUWorker.hpp"

#include <algorithm>
#include <cmath>

using namespace ANN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreCPUWorker<T>::CoreCPUWorker(const LayersConfig& layersConfig, const TrainingConfig<T>& trainingConfig,
                                const Parameters<T>& parameters, const CostFunctionConfig<T>& costFunctionConfig,
                                bool allocateTrainingBuffers)
  : layersConfig(layersConfig),
    trainingConfig(trainingConfig),
    parameters(parameters)
{
  this->costFunctionConfig = costFunctionConfig;
  this->allocate(allocateTrainingBuffers);
}

//===================================================================================================================//
//-- Allocation --//
//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::allocate(bool allocateTrainingBuffers)
{
  ulong numLayers = this->layersConfig.size();

  this->actvs.resize(numLayers);
  this->zs.resize(numLayers);

  for (ulong l = 0; l < numLayers; l++) {
    ulong numNeurons = this->layersConfig[l].numNeurons;
    this->actvs[l].resize(numNeurons);
  }

  for (ulong l = 1; l < numLayers; l++) {
    ulong numNeurons = this->layersConfig[l].numNeurons;
    this->zs[l].resize(numNeurons);
  }

  if (!allocateTrainingBuffers)
    return;

  this->dCost_dActvs.resize(numLayers);
  this->dCost_dWeights.resize(numLayers);
  this->dCost_dBiases.resize(numLayers);
  this->accum_dCost_dWeights.resize(numLayers);
  this->accum_dCost_dBiases.resize(numLayers);
  this->dropoutMasks.resize(numLayers);

  // Allocate dCost_dActvs for input layer (needed for backpropagateAndReturnInputGradients)
  this->dCost_dActvs[0].resize(this->layersConfig[0].numNeurons);

  for (ulong l = 1; l < numLayers; l++) {
    ulong numNeurons = this->layersConfig[l].numNeurons;
    ulong prevNumNeurons = this->actvs[l - 1].size();

    this->dCost_dActvs[l].resize(numNeurons);
    this->dCost_dWeights[l].resize(numNeurons);
    this->dCost_dBiases[l].resize(numNeurons);
    this->accum_dCost_dWeights[l].resize(numNeurons);
    this->accum_dCost_dBiases[l].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; j++) {
      this->dCost_dWeights[l][j].resize(prevNumNeurons);
      this->accum_dCost_dWeights[l][j].resize(prevNumNeurons);
    }
  }
}

//===================================================================================================================//
//-- Forward pass --//
//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::propagate(const Input<T>& input, bool applyDropout)
{
  ulong numLayers = this->layersConfig.size();
  float dropoutRate = this->trainingConfig.dropoutRate;

  this->actvs[0] = input;

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      this->zs[l][j] = this->parameters.biases[l][j];

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->zs[l][j] += this->parameters.weights[l][j][k] * this->actvs[l - 1][k];
      }
    }

    const T* zsData = this->zs[l].data();
    T* actvsData = this->actvs[l].data();
    ActvFuncType actvFuncType = layer.actvFuncType;

    ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, false);

    if (applyDropout && l < numLayers - 1 && dropoutRate > 0.0f) {
      T scale = static_cast<T>(1) / (static_cast<T>(1) - static_cast<T>(dropoutRate));
      std::bernoulli_distribution dist(1.0 - static_cast<double>(dropoutRate));

      this->dropoutMasks[l].resize(numNeurons);

      for (ulong j = 0; j < numNeurons; j++) {
        T mask = dist(this->rng) ? scale : static_cast<T>(0);
        this->dropoutMasks[l][j] = mask;
        this->actvs[l][j] *= mask;
      }
    }
  }
}

//===================================================================================================================//
//-- Output access --//
//===================================================================================================================//

template <typename T>
Output<T> CoreCPUWorker<T>::getOutput() const
{
  ulong numLayers = this->layersConfig.size();
  return this->actvs[numLayers - 1];
}

//===================================================================================================================//
//-- Loss --//
//===================================================================================================================//

template <typename T>
T CoreCPUWorker<T>::computeLoss(const Output<T>& expected)
{
  return this->calculateLoss(this->getOutput(), expected);
}

//===================================================================================================================//
//-- Backward pass --//
//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::backpropagate(const Output<T>& output)
{
  ulong numLayers = this->layersConfig.size();

  // For the last layer, calculate dCost_dActv (no dropout on output layer)
  ulong l = numLayers - 1;

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;

  for (ulong j = 0; j < numNeurons; j++) {
    this->dCost_dActvs[l][j] = this->calc_dCost_dActv(j, output);
  }

  for (ulong j = 0; j < numNeurons; j++) {
    this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    for (ulong k = 0; k < prevNumNeurons; k++) {
      this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
    }
  }

  // For the remaining layers, calculate backwards
  for (ulong l = numLayers - 2; l >= 1; l--) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // First pass: compute all dCost_dActvs for this layer
    for (ulong j = 0; j < numNeurons; j++) {
      this->dCost_dActvs[l][j] = this->calc_dCost_dActv(l, j);
    }

    // Apply dropout mask to gradients (same mask as forward pass)
    if (!this->dropoutMasks[l].empty()) {
      for (ulong j = 0; j < numNeurons; j++) {
        this->dCost_dActvs[l][j] *= this->dropoutMasks[l][j];
      }
    }

    // Second pass: compute bias and weight gradients
    for (ulong j = 0; j < numNeurons; j++) {
      this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
      }
    }

    if (l == 1)
      break;
  }
}

//===================================================================================================================//

template <typename T>
Tensor1D<T> CoreCPUWorker<T>::backpropagateAndReturnInputGradients(const Output<T>& output)
{
  this->backpropagate(output);

  // Additionally compute dCost_dActvs for the input layer (layer 0)
  // This is needed by external orchestrators (e.g., CNN) to continue backpropagation
  ulong inputNumNeurons = this->layersConfig[0].numNeurons;

  for (ulong k = 0; k < inputNumNeurons; k++) {
    this->dCost_dActvs[0][k] = this->calc_dCost_dActv(0, k);
  }

  return this->dCost_dActvs[0];
}

//===================================================================================================================//
//-- Gradient helpers --//
//===================================================================================================================//

// Particular case for the last layer.
template <typename T>
T CoreCPUWorker<T>::calc_dCost_dActv(ulong j, const Output<T>& output)
{
  ulong numLayers = this->layersConfig.size();
  ulong l = numLayers - 1;

  T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[j] : static_cast<T>(1);

  switch (this->costFunctionConfig.type) {
  case CostFunctionType::CROSS_ENTROPY: {
    // Cross-entropy: dL/da_j = -y_j / a_j (with epsilon for numerical stability)
    const T epsilon = static_cast<T>(1e-7);
    T pred = std::max(this->actvs[l][j], epsilon);
    return -weight * output[j] / pred;
  }

  case CostFunctionType::SQUARED_DIFFERENCE:
  case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
  default:
    // Squared difference: dL/da_j = 2 * w * (a_j - y_j)
    return 2 * weight * (this->actvs[l][j] - output[j]);
  }
}

//===================================================================================================================//

template <typename T>
T CoreCPUWorker<T>::calc_dCost_dActv(ulong l, ulong k)
{
  const Layer& nextLayer = this->layersConfig[l + 1];
  ulong nextNumNeurons = nextLayer.numNeurons;
  ActvFuncType actvFuncType = nextLayer.actvFuncType;

  const T* zsData = this->zs[l + 1].data();
  T* actvsData = const_cast<T*>(this->actvs[l + 1].data());
  const T* dCost_dActvsData = this->dCost_dActvs[l + 1].data();

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
T CoreCPUWorker<T>::calc_dCost_dWeight(ulong l, ulong j, ulong k)
{
  T actv = this->actvs[l - 1][k];

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;
  ActvFuncType actvFuncType = layer.actvFuncType;

  const T* zsData = this->zs[l].data();
  T* actvsData = const_cast<T*>(this->actvs[l].data());
  const T* dCost_dActvsData = this->dCost_dActvs[l].data();

  Tensor1D<T> dCost_dZs(numNeurons);
  T* dCost_dZsData = dCost_dZs.data();

  ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, true, dCost_dActvsData, dCost_dZsData);

  T dCost_dZ = dCost_dZs[j];

  return actv * dCost_dZ;
}

//===================================================================================================================//

template <typename T>
T CoreCPUWorker<T>::calc_dCost_dBias(ulong l, ulong j)
{
  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;
  ActvFuncType actvFuncType = layer.actvFuncType;

  const T* zsData = this->zs[l].data();
  T* actvsData = const_cast<T*>(this->actvs[l].data());
  const T* dCost_dActvsData = this->dCost_dActvs[l].data();

  Tensor1D<T> dCost_dZs(numNeurons);
  T* dCost_dZsData = dCost_dZs.data();

  ActvFunc::calculate(zsData, actvsData, numNeurons, actvFuncType, true, dCost_dActvsData, dCost_dZsData);

  T dCost_dZ = dCost_dZs[j];

  return dCost_dZ;
}

//===================================================================================================================//
//-- Accumulation --//
//===================================================================================================================//

template <typename T>
void CoreCPUWorker<T>::accumulate()
{
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

template <typename T>
void CoreCPUWorker<T>::resetAccumulators()
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
// Explicit template instantiations.
template class ANN::CoreCPUWorker<int>;
template class ANN::CoreCPUWorker<double>;
template class ANN::CoreCPUWorker<float>;
