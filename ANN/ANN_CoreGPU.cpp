#include "ANN_CoreGPU.hpp"
#include "ANN_Utils.hpp"
\
#include <iostream>

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

// oclwCore(false) is in the member initializer list to avoid double OpenCLWrapper::Core initialization.
template <typename T>
CoreGPU<T>::CoreGPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig), oclwCore(false) {
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
Output<T> CoreGPU<T>::run(const Input<T>& input) {
  this->writeInput(input);
  this->propagate(input);
  Output<T> output = this->readOutput();

  return output;
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::train(const Samples<T>& samples) {
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
void CoreGPU<T>::allocateCommon() {
  ulong numLayers = this->layersConfig.size();

  this->parameters.weights.resize(numLayers);
  this->parameters.biases.resize(numLayers);

  for (ulong l = 1; l < numLayers; l++) {
    Layer layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // Get the layer l activation function type and set it.
    this->parameters.weights[l].resize(numNeurons);
    this->parameters.biases[l].resize(numNeurons);

    // The number of neurons is the same as the number of activations.
    ulong prevNumNeurons = this->actvs[l - 1].size();

    for (ulong j = 0; j < numNeurons; j++) {
      this->parameters.weights[l][j].resize(prevNumNeurons);
    }
  }

  std::cout << "Loading OpenCL kernel...";
  this->oclwCore.addSourceFile("../Kernels.cpp.cl");
  std::cout << "OpenCL kernel loaded.";

  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();

  std::cout << "Allocating common buffers...";
  this->oclwCore. template allocateBuffer<T>("actvs", totalNumNeurons);
  this->oclwCore. template allocateBuffer<T>("weights", Utils<T>::count(this->parameters.weights));
  this->oclwCore. template allocateBuffer<T>("biases", Utils<T>::count(this->parameters.biases));
  this->oclwCore. template allocateBuffer<T>("zs", totalNumNeurons);
  std::cout << "Common buffers allocation done.";
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::allocateTraining() {
  ulong totalNumWeights = Utils<T>::count(this->parameters.weights);
  ulong totalNumBiases = Utils<T>::count(this->parameters.biases);

  std::cout << "Allocating training buffers...";
  this->oclwCore. template allocateBuffer<T>("dCost_dWeights", totalNumWeights);
  this->oclwCore. template allocateBuffer<T>("accum_dCost_dWeights", totalNumWeights);

  this->oclwCore. template allocateBuffer<T>("dCost_dBiases", totalNumBiases);
  this->oclwCore. template allocateBuffer<T>("accum_dCost_dBiases", totalNumBiases);
  std::cout << "Training buffers allocation done.";
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::propagate(const Input<T>& input) {
  ulong numLayers = this->layersConfig.size();

  // Set the actvs values of the Neurons of the first layer the same values as the input.
  this->oclwCore. template writeBuffer<T>("actvs", input, 0);

  ulong offset = 0;

  // Propagate from the second layer on, as the first layer is input only.
  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    this->oclwCore.clearKernels();

    this->oclwCore.addKernel("calculate_zs", numNeurons, offset);
    this->oclwCore. template addArgument<T>("calculate_zs", "zs");
    this->oclwCore. template addArgument<T>("calculate_zs", "weights");
    this->oclwCore. template addArgument<T>("calculate_zs", "actvs");
    this->oclwCore. template addArgument<ulong>("calculate_zs", l);
    this->oclwCore. template addArgument<LayersConfig>("calculate_zs", this->layersConfig);
    this->oclwCore. template addArgument<ulong>("calculate_zs", this->layersConfig.size());

    this->oclwCore.addKernel("calculate_actvs", numNeurons, offset);
    this->oclwCore. template addArgument<T>("calculate_zs", "actvs");
    this->oclwCore. template addArgument<T>("calculate_zs", "zs");
    this->oclwCore. template addArgument<ulong>("calculate_zs", l);
    this->oclwCore. template addArgument<LayersConfig>("calculate_zs", this->layersConfig);
    this->oclwCore. template addArgument<ulong>("calculate_zs", this->layersConfig.size());

    this->oclwCore.run();

    offset += numNeurons;
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::backpropagate(const Output<T>& output) {
  ulong numLayers = this->layersConfig.size();

  // For the last layer, calculate dCost_dActv
  ulong l = numLayers - 1;

  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;

  ulong numActvs = numNeurons;
  ulong numBiases = numNeurons;
  ulong numWeights = Utils<T>::count(this->parameters.weights[l]);

  ulong offsetActvs = this->layersConfig.getTotalNumNeurons() - numActvs;
  ulong offsetBiases = this->layersConfig.getTotalNumNeurons() - numBiases;
  ulong offsetWeights = Utils<T>::count(this->parameters.weights) - numWeights;

  this->oclwCore.clearKernels();

  this->oclwCore.addKernel("calculate_dCost_dActv_last_layer", numActvs, offsetActvs);
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "dCost_dAcvts");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "acvts");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "output");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", output.size());
  this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dActv_last_layer", this->layersConfig);
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", this->layersConfig.size());

  this->oclwCore.addKernel("calculate_dCost_dBias", numBiases, offsetBiases);
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dBiases");
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "zs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", l);
  this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dBias", this->layersConfig);

  this->oclwCore.addKernel("calculate_dCost_dWeight", numWeights, offsetWeights);
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dWeights");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "acvts");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "zs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", l);
  this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dWeight", this->layersConfig);
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", this->layersConfig.size());

  this->oclwCore.run();

  for (ulong l = numLayers - 1; l >= 1; l--) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    ulong numActvs = numNeurons;
    ulong numBiases = numNeurons;
    ulong numWeights = Utils<T>::count(this->parameters.weights[l]);

    offsetActvs -= numActvs;
    offsetBiases -= numBiases;
    offsetWeights -= numWeights;

    this->oclwCore.clearKernels();

    this->oclwCore.addKernel("calculate_dCost_dActv", numActvs, offsetActvs);
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "dCost_dAcvts");
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "acvts");
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "weights");
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "zs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv", l);
    this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dActv", this->layersConfig);
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv", this->layersConfig.size());

    this->oclwCore.addKernel("calculate_dCost_dBias", numBiases, offsetBiases);
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dBiases");
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "zs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", l);
    this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dBias", this->layersConfig);

    this->oclwCore.addKernel("calculate_dCost_dWeight", numWeights, offsetWeights);
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dWeights");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "acvts");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "zs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", l);
    this->oclwCore. template addArgument<LayersConfig>("calculate_dCost_dWeight", this->layersConfig);
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", this->layersConfig.size());

    this->oclwCore.run();
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::accumulate() {
  this->oclwCore.clearKernels();

  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  this->oclwCore.clearKernels();

  this->oclwCore.addKernel("accumulate_dCost_dBiases", numBiases, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "accum_dCost_dBiases");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "dCost_dBiases");

  this->oclwCore.addKernel("accumulate_dCost_dWeights", numWeights, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "accum_dCost_dWeights");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "dCost_dWeights");

  this->oclwCore.run();
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::update(ulong numSamples) {
  this->oclwCore.clearKernels();

  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  this->oclwCore.clearKernels();

  this->oclwCore.addKernel("update_dCost_dBiases", numBiases, 0);
  this->oclwCore. template addArgument<T>("update_dCost_dBiases", "dCost_dBiases");
  this->oclwCore. template addArgument<T>("update_dCost_dBiases", "accum_dCost_dBiases");
  this->oclwCore. template addArgument<ulong>("update_dCost_dBiases", numSamples);
  this->oclwCore. template addArgument<float>("update_dCost_dBiases", this->trainingConfig.learningRate);

  this->oclwCore.addKernel("update_dCost_dBWeights", numWeights, 0);
  this->oclwCore. template addArgument<T>("update_dCost_dBWeights", "dCost_dWeights");
  this->oclwCore. template addArgument<T>("update_dCost_dBWeights", "accum_dCost_dWeights");
  this->oclwCore. template addArgument<ulong>("update_dCost_dBWeights", numSamples);
  this->oclwCore. template addArgument<float>("update_dCost_dBWeights", this->trainingConfig.learningRate);

  this->oclwCore.run();
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::writeInput(const Input<T>& input) {
  ulong numNeurons = this->layersConfig[0].numNeurons;

  //this->oclwCore->
}

//===================================================================================================================//

template <typename T>
Output<T> CoreGPU<T>::readOutput() {
  ulong numLayers = this->layersConfig.size();

  ulong outputOffset = 0;

  for (ulong l = 0; l < numLayers - 1; l++) {
    outputOffset += this->layersConfig[l].numNeurons;
  }

  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();
  ulong outputNumNeurons = totalNumNeurons - outputOffset;

  Output<T> output;
  output.resize(outputNumNeurons);

  this->oclwCore.readBuffer("actvs", output, outputOffset);

  return output;
}

//===================================================================================================================//
// (Optional) Explicit template instantiations.
template class ANN::CoreGPU<int>;
template class ANN::CoreGPU<double>;
template class ANN::CoreGPU<float>;
