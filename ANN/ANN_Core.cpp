#include "ANN_Core.hpp"

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) {
  this->init(coreConfig);
}

//===================================================================================================================//

template <typename T>
void Core<T>::init(const CoreConfig<T>& coreConfig) {
  this->coreModeType = coreConfig.coreModeType;

  this->layersConfig = coreConfig.layersConfig;
  this->trainingConfig = coreConfig.trainingConfig;
  this->parameters = coreConfig.parameters;

  this->sanityCheck();

  this->allocateCommon();

  switch (this->coreModeType) {
    case CoreModeType::TRAIN:
      this->allocateTraining();
      break;
    case CoreModeType::RUN:
    case CoreModeType::UNKNOWN:
      break;
  }

  if (this->coreModeType == CoreModeType::TRAIN) {
    this->allocateTraining();
  }
}

//===================================================================================================================//

template <typename T>
Output<T> Core<T>::run(const Input<T>& input) {
  this->propagate(input);
  Output<T> output = this->getOutput();

  return output;
}

//===================================================================================================================//

template <typename T>
void Core<T>::train(const Samples<T>& samples) {
  uint numSamples = samples.size();

  uint numEpochs = this->trainingConfig.numEpochs;

  for (uint e = 0; e < numEpochs; e++) {
    for (uint s = 0; s < numSamples; s++) {
      const Input<T>& input = samples[s].input;
      const Output<T>& output = samples[s].output;

      this->propagate(input);
      this->backpropagate(output);
      this->accumulate();
    }

    this->update(numSamples);
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck() {
  if (this->coreModeType == CoreModeType::UNKNOWN) {
    throw std::runtime_error("Unkown coreModeType.");
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::allocateCommon() {
  uint numLayers = this->layersConfig.size();

  this->actvs.resize(numLayers);

  for (uint l = 0; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    this->actvs[l].resize(numNeurons);
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::allocateTraining() {
  uint numLayers = this->layersConfig.size();

  this->parameters.weights.resize(numLayers);
  this->parameters.biases.resize(numLayers);
  this->zs.resize(numLayers);

  this->dCost_dActvs.resize(numLayers);
  this->accum_dCost_dWeights.resize(numLayers);

  this->dCost_dWeights.resize(numLayers);
  this->accum_dCost_dBiases.resize(numLayers);

  for (uint l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    // Get the layer l activation function type and set it.
    this->parameters.weights[l].resize(numNeurons);
    this->parameters.biases[l].resize(numNeurons);
    this->zs[l].resize(numNeurons);

    this->dCost_dWeights[l].resize(numNeurons);
    this->accum_dCost_dWeights[l].resize(numNeurons);

    this->dCost_dBiases[l].resize(numNeurons);
    this->accum_dCost_dBiases[l].resize(numLayers);

    // The number of neurons is the same as the number of activations.
    uint prevNumNeurons = this->actvs[l - 1].size();

    for (uint j = 0; j < numNeurons; j++) {
      this->parameters.weights[l][j].resize(prevNumNeurons);

      this->dCost_dWeights[l][j].resize(prevNumNeurons);
      this->accum_dCost_dWeights[l][j].resize(prevNumNeurons);
    }
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::propagate(const Input<T>& input) {
  uint numLayers = this->layersConfig.size();

  // Set the actvs values of the Neurons of the first layer the same values as the input.
  this->actvs[0] = input;

  // Propagate from the second layer on, as the first layer is input only.
  for (uint l = 1; l < numLayers; l++) {
    const Layer& prevLayer = this->layersConfig[l - 1];
    uint prevNumNeurons = prevLayer.numNeurons;

    const Layer& layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    for (uint j = 0; j < numNeurons; j++) {
      this->zs[l][j] = 0;

      for (uint k = 0; k < prevNumNeurons; k++) {
        this->zs[l][j] += this->parameters.weights[l][j][k] * this->actvs[l - 1][k];
      }

      ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
      this->actvs[l][j] = ActvFunc::calculate(this->zs[l][j], actvFuncType);
    }
  }
}

//===================================================================================================================//

template <typename T>
Output<T> Core<T>::getOutput() {
  Output<T> output;

  return output;
}

//===================================================================================================================//

template <typename T>
void Core<T>::backpropagate(const Output<T>& output) {
  uint numLayers = this->layersConfig.size();

  // For the last layer, calculate dCost_dActv
  uint l = numLayers - 1;

  const Layer& layer = this->layersConfig[l];
  uint numNeurons = layer.numNeurons;

  for (uint j = 0; j < numNeurons; j++) {
    this->dCost_dActvs[l][j] = this->calc_dCost_dActv(j, output);
    this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

    const Layer& prevLayer = this->layersConfig[l - 1];
    uint prevNumNeurons = prevLayer.numNeurons;

    for (uint k = 0; k < prevNumNeurons - 1; k++) {
      this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
    }
  }

  // For the remaining layers, calculate backwards
  for (uint l = numLayers - 1; l >= 1; l--) {
    const Layer& layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    for (uint j = 0; j < numNeurons - 1; j++) {
      // First we need to compute the
      this->dCost_dActvs[l][j] = this->calc_dCost_dActv(l, j);
      this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

      const Layer& prevLayer = this->layersConfig[l - 1];
      uint prevNumNeurons = prevLayer.numNeurons;

      for (uint k = 0; k < prevNumNeurons; k++) {
        this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::accumulate() {
  uint numLayers = this->layersConfig.size();

  for (uint l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    for (uint j = 0; j < numNeurons - 1; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      uint prevNumNeurons = prevLayer.numNeurons;

      this->accum_dCost_dBiases[l][j] += this->dCost_dBiases[l][j];

      for (uint k = 0; k < prevNumNeurons; k++) {
        this->accum_dCost_dWeights[l][j][k] += this->dCost_dWeights[l][j][k];
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::update(uint numSamples) {
  T learningRate = this->trainingConfig.learningRate;
  uint numLayers = this->layersConfig.size();

  for (uint l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    uint numNeurons = layer.numNeurons;

    for (uint j = 0; j < numNeurons - 1; j++) {
      const Layer& prevLayer = this->layersConfig[l - 1];
      uint prevNumNeurons = prevLayer.numNeurons;

      this->dCost_dBiases[l][j] += learningRate * (this->accum_dCost_dBiases[l][j] / numSamples);

      for (uint k = 0; k < prevNumNeurons; k++) {
        this->dCost_dWeights[l][j][k] = learningRate * (this->accum_dCost_dWeights[l][j][k] / numSamples);
      }
    }
  }
}

//===================================================================================================================//

// Particular case for the last layer.
template <typename T>
T Core<T>::calc_dCost_dActv(uint j, const Output<T>& output) {
  uint numLayers = this->layersConfig.size();

  uint l = numLayers - 1;

  return 2 * (this->actvs[l][j] - output[j]);
}

//===================================================================================================================//

template <typename T>
T Core<T>::calc_dCost_dActv(uint l, uint k) {
  const Layer& nextLayer = this->layersConfig[l + 1];
  uint nextNumNeurons = nextLayer.numNeurons;

  T sum = 0;

  for (uint j = 0; j < nextNumNeurons; j++) {
    T weight = this->parameters.weights[l + 1][j][k];
    T z = this->zs[l + 1][j];

    ActvFuncType actvFuncType = this->layersConfig[l + 1].actvFuncType;
    T dActvFunc_z = ActvFunc::calculate(z, actvFuncType, true);

    T dCost_dActv = this->dCost_dActvs[l + 1][j];

    sum += weight * dActvFunc_z * dCost_dActv;
  }

  return sum;
}

//===================================================================================================================//

template <typename T>
T Core<T>::calc_dCost_dWeight(uint l, uint j, uint k) {
  T actv = this->actvs[l - 1][k];
  T z = this->zs[l][j];

  ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
  T dActvFunc_z = ActvFunc::calculate(z, actvFuncType, true);

  T dCost_dActv = this->dCost_dActvs[l][j];

  return actv * dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

template <typename T>
T Core<T>::calc_dCost_dBias(uint l, uint j) {
  T z = this->zs[l][j];

  ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
  T dActvFunc_z = ActvFunc::calculate(z, actvFuncType, true);

  return dActvFunc_z * this->dCost_dActvs[l][j];
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::Core<int>;
template class ANN::Core<double>;
template class ANN::Core<float>;
