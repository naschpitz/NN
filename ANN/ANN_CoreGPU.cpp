#include "ANN_CoreGPU.hpp"
#include "ANN_Utils.hpp"

#include <cmath>
#include <iostream>
#include <random>

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

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

  // Set up all kernels once before training loop
  if (!this->sampleKernelsSetup) {
    this->setupSampleKernels();
    this->sampleKernelsSetup = true;
  }

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = 0;

    // Reset accumulators at the start of each epoch
    this->resetAccumulators();

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
T CoreGPU<T>::calculateLoss(const Output<T>& expected) {
  // Read output activations from GPU
  Output<T> actual = this->readOutput();

  T loss = 0;
  for (ulong i = 0; i < expected.size(); i++) {
    T diff = actual[i] - expected[i];
    loss += diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::allocateCommon() {
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

  std::cout << "Loading OpenCL kernels...\n";
  // Load source files in order - they will be concatenated by OpenCL
  this->oclwCore.addSourceFile("extern/ANN/Defines.hpp.cl");
  this->oclwCore.addSourceFile("extern/ANN/ActvFunc.cpp.cl");
  this->oclwCore.addSourceFile("extern/ANN/IdxHelper.cpp.cl");
  this->oclwCore.addSourceFile("extern/ANN/Kernels.cpp.cl");
  std::cout << "OpenCL kernels loaded.\n";

  ulong totalNumNeurons = this->layersConfig.getTotalNumNeurons();

  std::cout << "Allocating common buffers...";
  this->oclwCore. template allocateBuffer<T>("actvs", totalNumNeurons);
  this->oclwCore. template allocateBuffer<T>("weights", Utils<T>::count(this->parameters.weights));
  this->oclwCore. template allocateBuffer<T>("biases", Utils<T>::count(this->parameters.biases));
  this->oclwCore. template allocateBuffer<T>("zs", totalNumNeurons);
  this->oclwCore. template allocateBuffer<T>("dCost_dActvs", totalNumNeurons);

  // Allocate buffer for layers configuration (used by kernels)
  // Each Layer is: ulong numNeurons (8 bytes) + ActvFuncType (4 bytes) + padding (4 bytes) = 16 bytes
  this->oclwCore. template allocateBuffer<Layer>("layers", numLayers);

  // Write layers configuration to GPU
  std::vector<Layer> layersVec(this->layersConfig.begin(), this->layersConfig.end());
  this->oclwCore. template writeBuffer<Layer>("layers", layersVec, 0);

  std::cout << "Common buffers allocation done.";

  // Write initialized weights and biases to GPU buffers
  std::vector<T> flatWeights = Utils<T>::flatten(this->parameters.weights);
  std::vector<T> flatBiases = Utils<T>::flatten(this->parameters.biases);
  this->oclwCore. template writeBuffer<T>("weights", flatWeights, 0);
  this->oclwCore. template writeBuffer<T>("biases", flatBiases, 0);
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
void CoreGPU<T>::setupSampleKernels() {
  ulong numLayers = this->layersConfig.size();

  std::cout << "Setting up sample kernels (propagate + backpropagate + accumulate)...\n";

  // Clear any existing kernels
  this->oclwCore.clearKernels();

  // === PROPAGATE KERNELS ===
  // Add kernels for all layers (layer 1 to N-1)
  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    // Create unique IDs for this layer's kernels
    std::string calculate_zs_id = "calculate_zs_layer" + std::to_string(l);
    std::string calculate_actvs_id = "calculate_actvs_layer" + std::to_string(l);

    // calculate_zs kernel: computes weighted sum + bias for each neuron
    this->oclwCore.addKernel(calculate_zs_id, "calculate_zs", numNeurons, 0);
    this->oclwCore. template addArgument<T>(calculate_zs_id, "zs");
    this->oclwCore. template addArgument<T>(calculate_zs_id, "weights");
    this->oclwCore. template addArgument<T>(calculate_zs_id, "actvs");
    this->oclwCore. template addArgument<T>(calculate_zs_id, "biases");
    this->oclwCore. template addArgument<ulong>(calculate_zs_id, l);
    this->oclwCore. template addArgument<Layer>(calculate_zs_id, "layers");
    this->oclwCore. template addArgument<ulong>(calculate_zs_id, numLayers);

    // calculate_actvs kernel: applies activation function
    this->oclwCore.addKernel(calculate_actvs_id, "calculate_actvs", numNeurons, 0);
    this->oclwCore. template addArgument<T>(calculate_actvs_id, "actvs");
    this->oclwCore. template addArgument<T>(calculate_actvs_id, "zs");
    this->oclwCore. template addArgument<ulong>(calculate_actvs_id, l);
    this->oclwCore. template addArgument<Layer>(calculate_actvs_id, "layers");
    this->oclwCore. template addArgument<ulong>(calculate_actvs_id, numLayers);
  }

  // === BACKPROPAGATE KERNELS ===
  // Last layer kernels
  ulong l = numLayers - 1;
  const Layer& lastLayer = this->layersConfig[l];
  ulong numNeurons = lastLayer.numNeurons;
  ulong numBiases = numNeurons;
  ulong numWeights = Utils<T>::count(this->parameters.weights[l]);

  // Allocate outputs buffer for backpropagation
  this->oclwCore. template allocateBuffer<T>("outputs", this->layersConfig[numLayers - 1].numNeurons);

  // calculate_dCost_dActv_last_layer (unique, only one instance)
  this->oclwCore.addKernel("calculate_dCost_dActv_last_layer", numNeurons, 0);
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "dCost_dActvs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "actvs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "outputs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", this->layersConfig[numLayers - 1].numNeurons);
  this->oclwCore. template addArgument<Layer>("calculate_dCost_dActv_last_layer", "layers");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", numLayers);

  // Create unique IDs for last layer's kernels
  std::string dCost_dBias_last_id = "calculate_dCost_dBias_layer" + std::to_string(l);
  std::string dCost_dWeight_last_id = "calculate_dCost_dWeight_layer" + std::to_string(l);

  // calculate_dCost_dBias for last layer
  this->oclwCore.addKernel(dCost_dBias_last_id, "calculate_dCost_dBias", numBiases, 0);
  this->oclwCore. template addArgument<T>(dCost_dBias_last_id, "dCost_dBiases");
  this->oclwCore. template addArgument<T>(dCost_dBias_last_id, "zs");
  this->oclwCore. template addArgument<T>(dCost_dBias_last_id, "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>(dCost_dBias_last_id, l);
  this->oclwCore. template addArgument<Layer>(dCost_dBias_last_id, "layers");
  this->oclwCore. template addArgument<ulong>(dCost_dBias_last_id, numLayers);

  // calculate_dCost_dWeight for last layer
  this->oclwCore.addKernel(dCost_dWeight_last_id, "calculate_dCost_dWeight", numWeights, 0);
  this->oclwCore. template addArgument<T>(dCost_dWeight_last_id, "dCost_dWeights");
  this->oclwCore. template addArgument<T>(dCost_dWeight_last_id, "actvs");
  this->oclwCore. template addArgument<T>(dCost_dWeight_last_id, "zs");
  this->oclwCore. template addArgument<T>(dCost_dWeight_last_id, "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>(dCost_dWeight_last_id, l);
  this->oclwCore. template addArgument<Layer>(dCost_dWeight_last_id, "layers");
  this->oclwCore. template addArgument<ulong>(dCost_dWeight_last_id, numLayers);

  // Backpropagate through hidden layers (from second-to-last to first hidden layer)
  for (ulong layer_idx = numLayers - 2; layer_idx >= 1; layer_idx--) {
    const Layer& curr_layer = this->layersConfig[layer_idx];
    ulong curr_numNeurons = curr_layer.numNeurons;
    ulong curr_numBiases = curr_numNeurons;
    ulong curr_numWeights = Utils<T>::count(this->parameters.weights[layer_idx]);

    // Create unique IDs for this layer's kernels
    std::string dCost_dActv_id = "calculate_dCost_dActv_layer" + std::to_string(layer_idx);
    std::string dCost_dBias_id = "calculate_dCost_dBias_layer" + std::to_string(layer_idx);
    std::string dCost_dWeight_id = "calculate_dCost_dWeight_layer" + std::to_string(layer_idx);

    // calculate_dCost_dActv
    this->oclwCore.addKernel(dCost_dActv_id, "calculate_dCost_dActv", curr_numNeurons, 0);
    this->oclwCore. template addArgument<T>(dCost_dActv_id, "dCost_dActvs");
    this->oclwCore. template addArgument<T>(dCost_dActv_id, "weights");
    this->oclwCore. template addArgument<T>(dCost_dActv_id, "zs");
    this->oclwCore. template addArgument<ulong>(dCost_dActv_id, layer_idx);
    this->oclwCore. template addArgument<Layer>(dCost_dActv_id, "layers");
    this->oclwCore. template addArgument<ulong>(dCost_dActv_id, numLayers);

    // calculate_dCost_dBias
    this->oclwCore.addKernel(dCost_dBias_id, "calculate_dCost_dBias", curr_numBiases, 0);
    this->oclwCore. template addArgument<T>(dCost_dBias_id, "dCost_dBiases");
    this->oclwCore. template addArgument<T>(dCost_dBias_id, "zs");
    this->oclwCore. template addArgument<T>(dCost_dBias_id, "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>(dCost_dBias_id, layer_idx);
    this->oclwCore. template addArgument<Layer>(dCost_dBias_id, "layers");
    this->oclwCore. template addArgument<ulong>(dCost_dBias_id, numLayers);

    // calculate_dCost_dWeight
    this->oclwCore.addKernel(dCost_dWeight_id, "calculate_dCost_dWeight", curr_numWeights, 0);
    this->oclwCore. template addArgument<T>(dCost_dWeight_id, "dCost_dWeights");
    this->oclwCore. template addArgument<T>(dCost_dWeight_id, "actvs");
    this->oclwCore. template addArgument<T>(dCost_dWeight_id, "zs");
    this->oclwCore. template addArgument<T>(dCost_dWeight_id, "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>(dCost_dWeight_id, layer_idx);
    this->oclwCore. template addArgument<Layer>(dCost_dWeight_id, "layers");
    this->oclwCore. template addArgument<ulong>(dCost_dWeight_id, numLayers);

    // Break condition for ulong loop (can't go negative)
    if (layer_idx == 1) break;
  }

  // === ACCUMULATE KERNELS ===
  ulong totalNumBiases = Utils<T>::count(this->parameters.biases);
  ulong totalNumWeights = Utils<T>::count(this->parameters.weights);

  this->oclwCore.addKernel("accumulate_dCost_dBiases", totalNumBiases, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "accum_dCost_dBiases");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "dCost_dBiases");
  this->oclwCore. template addArgument<ulong>("accumulate_dCost_dBiases", totalNumBiases);

  this->oclwCore.addKernel("accumulate_dCost_dWeights", totalNumWeights, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "accum_dCost_dWeights");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "dCost_dWeights");
  this->oclwCore. template addArgument<ulong>("accumulate_dCost_dWeights", totalNumWeights);

  std::cout << "Sample kernels setup done.\n";
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::propagate(const Input<T>& input) {
  // Write input to actvs buffer (first layer)
  this->oclwCore. template writeBuffer<T>("actvs", input, 0);

  // If kernels are set up, just run them
  // Otherwise, this is being called outside of train() (e.g., from run())
  if (this->sampleKernelsSetup) {
    // Kernels will be run at the end of sample processing (after backpropagate and accumulate)
    // This is handled by the train() loop
    return;
  }

  // Fallback for run() mode - set up and run kernels per layer (old behavior)
  ulong numLayers = this->layersConfig.size();
  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;

    this->oclwCore.clearKernels();

    this->oclwCore.addKernel("calculate_zs", numNeurons, 0);
    this->oclwCore. template addArgument<T>("calculate_zs", "zs");
    this->oclwCore. template addArgument<T>("calculate_zs", "weights");
    this->oclwCore. template addArgument<T>("calculate_zs", "actvs");
    this->oclwCore. template addArgument<T>("calculate_zs", "biases");
    this->oclwCore. template addArgument<ulong>("calculate_zs", l);
    this->oclwCore. template addArgument<Layer>("calculate_zs", "layers");
    this->oclwCore. template addArgument<ulong>("calculate_zs", numLayers);

    this->oclwCore.addKernel("calculate_actvs", numNeurons, 0);
    this->oclwCore. template addArgument<T>("calculate_actvs", "actvs");
    this->oclwCore. template addArgument<T>("calculate_actvs", "zs");
    this->oclwCore. template addArgument<ulong>("calculate_actvs", l);
    this->oclwCore. template addArgument<Layer>("calculate_actvs", "layers");
    this->oclwCore. template addArgument<ulong>("calculate_actvs", numLayers);

    this->oclwCore.run();
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::backpropagate(const Output<T>& output) {
  // Write expected output to buffer for kernel use
  this->oclwCore. template writeBuffer<T>("outputs", output, 0);

  // If kernels are set up, just write buffer - run() will be called by accumulate()
  if (this->sampleKernelsSetup) {
    return;
  }

  // Fallback for non-training mode (old behavior)
  ulong numLayers = this->layersConfig.size();

  this->oclwCore. template allocateBuffer<T>("outputs", output.size());

  ulong l = numLayers - 1;
  const Layer& layer = this->layersConfig[l];
  ulong numNeurons = layer.numNeurons;
  ulong numBiases = numNeurons;
  ulong numWeights = Utils<T>::count(this->parameters.weights[l]);

  this->oclwCore.clearKernels();

  this->oclwCore.addKernel("calculate_dCost_dActv_last_layer", numNeurons, 0);
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "dCost_dActvs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "actvs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dActv_last_layer", "outputs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", output.size());
  this->oclwCore. template addArgument<Layer>("calculate_dCost_dActv_last_layer", "layers");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv_last_layer", numLayers);

  this->oclwCore.addKernel("calculate_dCost_dBias", numBiases, 0);
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dBiases");
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "zs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", l);
  this->oclwCore. template addArgument<Layer>("calculate_dCost_dBias", "layers");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", numLayers);

  this->oclwCore.addKernel("calculate_dCost_dWeight", numWeights, 0);
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dWeights");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "actvs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "zs");
  this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dActvs");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", l);
  this->oclwCore. template addArgument<Layer>("calculate_dCost_dWeight", "layers");
  this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", numLayers);

  this->oclwCore.run();

  for (ulong layer_idx = numLayers - 2; layer_idx >= 1; layer_idx--) {
    const Layer& curr_layer = this->layersConfig[layer_idx];
    ulong curr_numNeurons = curr_layer.numNeurons;
    ulong curr_numBiases = curr_numNeurons;
    ulong curr_numWeights = Utils<T>::count(this->parameters.weights[layer_idx]);

    this->oclwCore.clearKernels();

    this->oclwCore.addKernel("calculate_dCost_dActv", curr_numNeurons, 0);
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "dCost_dActvs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "weights");
    this->oclwCore. template addArgument<T>("calculate_dCost_dActv", "zs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv", layer_idx);
    this->oclwCore. template addArgument<Layer>("calculate_dCost_dActv", "layers");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dActv", numLayers);

    this->oclwCore.addKernel("calculate_dCost_dBias", curr_numBiases, 0);
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dBiases");
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "zs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dBias", "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", layer_idx);
    this->oclwCore. template addArgument<Layer>("calculate_dCost_dBias", "layers");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dBias", numLayers);

    this->oclwCore.addKernel("calculate_dCost_dWeight", curr_numWeights, 0);
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dWeights");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "actvs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "zs");
    this->oclwCore. template addArgument<T>("calculate_dCost_dWeight", "dCost_dActvs");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", layer_idx);
    this->oclwCore. template addArgument<Layer>("calculate_dCost_dWeight", "layers");
    this->oclwCore. template addArgument<ulong>("calculate_dCost_dWeight", numLayers);

    this->oclwCore.run();

    if (layer_idx == 1) break;
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::accumulate() {
  // If kernels are set up, run all sample kernels (propagate + backpropagate + accumulate)
  if (this->sampleKernelsSetup) {
    this->oclwCore.run();
    return;
  }

  // Fallback for non-training mode (old behavior)
  this->oclwCore.clearKernels();

  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  this->oclwCore.addKernel("accumulate_dCost_dBiases", numBiases, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "accum_dCost_dBiases");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dBiases", "dCost_dBiases");
  this->oclwCore. template addArgument<ulong>("accumulate_dCost_dBiases", numBiases);

  this->oclwCore.addKernel("accumulate_dCost_dWeights", numWeights, 0);
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "accum_dCost_dWeights");
  this->oclwCore. template addArgument<T>("accumulate_dCost_dWeights", "dCost_dWeights");
  this->oclwCore. template addArgument<ulong>("accumulate_dCost_dWeights", numWeights);

  this->oclwCore.run();
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::setupUpdateKernels(ulong numSamples) {
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  std::cout << "Setting up update kernels...\n";

  // Clear sample kernels and set up update kernels
  this->oclwCore.clearKernels();

  this->oclwCore.addKernel("update_biases", numBiases, 0);
  this->oclwCore. template addArgument<T>("update_biases", "biases");
  this->oclwCore. template addArgument<T>("update_biases", "accum_dCost_dBiases");
  this->oclwCore. template addArgument<ulong>("update_biases", numSamples);
  this->oclwCore. template addArgument<float>("update_biases", this->trainingConfig.learningRate);
  this->oclwCore. template addArgument<ulong>("update_biases", numBiases);

  this->oclwCore.addKernel("update_weights", numWeights, 0);
  this->oclwCore. template addArgument<T>("update_weights", "weights");
  this->oclwCore. template addArgument<T>("update_weights", "accum_dCost_dWeights");
  this->oclwCore. template addArgument<ulong>("update_weights", numSamples);
  this->oclwCore. template addArgument<float>("update_weights", this->trainingConfig.learningRate);
  this->oclwCore. template addArgument<ulong>("update_weights", numWeights);

  std::cout << "Update kernels setup done.\n";
}

//===================================================================================================================//

template <typename T>
void CoreGPU<T>::update(ulong numSamples) {
  // Set up update kernels if not done yet (this clears sample kernels)
  if (!this->updateKernelsSetup) {
    this->setupUpdateKernels(numSamples);
    this->updateKernelsSetup = true;
  }

  // Run update kernels
  this->oclwCore.run();

  // After update, we need to re-setup sample kernels for next epoch
  // Reset the flag so they get set up again on first sample of next epoch
  this->sampleKernelsSetup = false;
  this->updateKernelsSetup = false;  // Reset so it gets set up with potentially new numSamples
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

template <typename T>
void CoreGPU<T>::resetAccumulators() {
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  // Create zero-filled vectors
  std::vector<T> zeroBiases(numBiases, static_cast<T>(0));
  std::vector<T> zeroWeights(numWeights, static_cast<T>(0));

  // Write zeros to GPU accumulator buffers
  this->oclwCore. template writeBuffer<T>("accum_dCost_dBiases", zeroBiases, 0);
  this->oclwCore. template writeBuffer<T>("accum_dCost_dWeights", zeroWeights, 0);
}

//===================================================================================================================//
// (Optional) Explicit template instantiations.
template class ANN::CoreGPU<int>;
template class ANN::CoreGPU<double>;
template class ANN::CoreGPU<float>;
