#include "ANN_CoreCPU.hpp"

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

template <typename T>
CoreCPU<T>::CoreCPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig) {
  this->allocateCommon();

  switch (this->coreModeType) {
    case CoreModeType::TRAIN:
      this->allocateTraining();
      break;
    case CoreModeType::RUN:
    case CoreModeType::UNKNOWN:
      break;
  }
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::run(const Input<T>& input) {
  this->propagate(input);
  Output<T> output = this->getOutput();

  return output;
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(const Samples<T>& samples) {
  ulong numSamples = samples.size();
  ulong numEpochs = this->trainingConfig.numEpochs;

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = 0;

    for (ulong s = 0; s < numSamples; s++) {
      const Input<T>& input = samples[s].input;
      const Output<T>& output = samples[s].output;

      this->propagate(input);

      T sampleLoss = this->calculateLoss(output);
      epochLoss += sampleLoss;

      // Call progress callback if set
      if (this->trainingCallback) {
        TrainingProgress<T> progress;
        progress.currentEpoch = e + 1;
        progress.totalEpochs = numEpochs;
        progress.currentSample = s + 1;
        progress.totalSamples = numSamples;
        progress.sampleLoss = sampleLoss;
        progress.epochLoss = 0;  // Not complete yet
        this->trainingCallback(progress);
      }

      this->backpropagate(output);
      this->accumulate();
    }

    this->update(numSamples);

    // Call callback with epoch completion (sample at max, epochLoss set)
    if (this->trainingCallback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = e + 1;
      progress.totalEpochs = numEpochs;
      progress.currentSample = numSamples;
      progress.totalSamples = numSamples;
      progress.sampleLoss = 0;
      progress.epochLoss = epochLoss / static_cast<T>(numSamples);
      this->trainingCallback(progress);
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateCommon() {
  ulong numLayers = this->layersConfig.size();

  this->actvs.resize(numLayers);
  this->parameters.weights.resize(numLayers);
  this->parameters.biases.resize(numLayers);
  this->zs.resize(numLayers);

  for (ulong l = 0; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    this->actvs[l].resize(numNeurons);
  }

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // Get the layer l activation function type and set it.
    this->parameters.weights[l].resize(numNeurons);
    this->parameters.biases[l].resize(numNeurons);
    this->zs[l].resize(numNeurons);

    // The number of neurons is the same as the number of activations.
    ulong prevNumNeurons = this->actvs[l - 1].size();

    for (ulong j = 0; j < numNeurons; j++) {
      this->parameters.weights[l][j].resize(prevNumNeurons);
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
void CoreCPU<T>::propagate(const Input<T>& input) {
  ulong numLayers = this->layersConfig.size();

  // Set the actvs values of the Neurons of the first layer the same values as the input.
  this->actvs[0] = input;

  // Propagate from the second layer on, as the first layer is input only.
  for (ulong l = 1; l < numLayers; l++) {
    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      this->zs[l][j] = 0;

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->zs[l][j] += this->parameters.weights[l][j][k] * this->actvs[l - 1][k];
      }

      ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
      this->actvs[l][j] = ActvFunc::calculate(this->zs[l][j], actvFuncType);
    }
  }
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::getOutput() {
  ulong numLayers = this->layersConfig.size();

  return this->actvs[numLayers - 1];
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::backpropagate(const Output<T>& output) {
  ulong numLayers = this->layersConfig.size();

  // For the last layer, calculate dCost_dActv
  ulong l = numLayers - 1;

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;

  for (ulong j = 0; j < numNeurons; j++) {
    this->dCost_dActvs[l][j] = this->calc_dCost_dActv(j, output);
    this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

    const Layer& prevLayer = this->layersConfig[l - 1];
    ulong prevNumNeurons = prevLayer.numNeurons;

    for (ulong k = 0; k < prevNumNeurons; k++) {
      this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
    }
  }

  // For the remaining layers, calculate backwards
  for (ulong l = numLayers - 1; l >= 1; l--) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    for (ulong j = 0; j < numNeurons; j++) {
      // First we need to compute the
      this->dCost_dActvs[l][j] = this->calc_dCost_dActv(l, j);
      this->dCost_dBiases[l][j] = this->calc_dCost_dBias(l, j);

      const Layer& prevLayer = this->layersConfig[l - 1];
      ulong prevNumNeurons = prevLayer.numNeurons;

      for (ulong k = 0; k < prevNumNeurons; k++) {
        this->dCost_dWeights[l][j][k] = this->calc_dCost_dWeight(l, j, k);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::accumulate() {
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

// Particular case for the last layer.
template <typename T>
T CoreCPU<T>::calc_dCost_dActv(ulong j, const Output<T>& output) {
  ulong numLayers = this->layersConfig.size();

  ulong l = numLayers - 1;

  return 2 * (this->actvs[l][j] - output[j]);
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calc_dCost_dActv(ulong l, ulong k) {
  const Layer& nextLayer = this->layersConfig[l + 1];
  ulong nextNumNeurons = nextLayer.numNeurons;

  T sum = 0;

  for (ulong j = 0; j < nextNumNeurons; j++) {
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
T CoreCPU<T>::calc_dCost_dWeight(ulong l, ulong j, ulong k) {
  T actv = this->actvs[l - 1][k];
  T z = this->zs[l][j];

  ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
  T dActvFunc_z = ActvFunc::calculate(z, actvFuncType, true);

  T dCost_dActv = this->dCost_dActvs[l][j];

  return actv * dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

template <typename T>
T CoreCPU<T>::calc_dCost_dBias(ulong l, ulong j) {
  T z = this->zs[l][j];

  ActvFuncType actvFuncType = this->layersConfig[l].actvFuncType;
  T dActvFunc_z = ActvFunc::calculate(z, actvFuncType, true);

  return dActvFunc_z * this->dCost_dActvs[l][j];
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::CoreCPU<int>;
template class ANN::CoreCPU<double>;
template class ANN::CoreCPU<float>;
