#include "ANN_GPUKernelBuilder.hpp"
#include "ANN_Utils.hpp"

#include <cmath>
#include <string>

using namespace ANN;

//===================================================================================================================//

template <typename T>
GPUKernelBuilder<T>::GPUKernelBuilder(OpenCLWrapper::Core* core, const LayersConfig& layersConfig,
                                      const Parameters<T>& parameters, const TrainingConfig<T>& trainingConfig,
                                      const CostFunctionConfig<T>& costFunctionConfig,
                                      GPUBufferManager<T>& bufferManager, LogLevel logLevel)
  : core(core),
    layersConfig(layersConfig),
    parameters(parameters),
    trainingConfig(trainingConfig),
    costFunctionConfig(costFunctionConfig),
    bufferManager(bufferManager),
    logLevel(logLevel)
{
}

//===================================================================================================================//
//-- Kernel setup --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupPredictKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupTrainingKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addBackpropagateKernels(false);
  this->addAccumulateKernels();

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupBackpropagateKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addBackpropagateKernels(true);

  this->backpropagateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupAccumulateKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addAccumulateKernels();

  this->accumulateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupUpdateKernels(ulong numSamples)
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addUpdateKernels(numSamples);

  this->updateKernelsSetup = true;
}

//===================================================================================================================//
//-- addUpdateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addUpdateKernels(ulong numSamples)
{
  ulong numBiases = Utils<T>::count(this->parameters.biases);
  ulong numWeights = Utils<T>::count(this->parameters.weights);

  if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
    const auto& opt = this->trainingConfig.optimizer;
    this->adam_t++;

    // Precompute bias corrections on CPU for numerical stability
    float bc1 = 1.0f - std::pow(static_cast<float>(opt.beta1), static_cast<float>(this->adam_t));
    float bc2 = 1.0f - std::pow(static_cast<float>(opt.beta2), static_cast<float>(this->adam_t));

    this->core->addKernel("update_biases_adam", numBiases, 0);
    this->core->template addArgument<T>("update_biases_adam", "biases");
    this->core->template addArgument<T>("update_biases_adam", "accum_dCost_dBiases");
    this->core->template addArgument<T>("update_biases_adam", "adam_m_biases");
    this->core->template addArgument<T>("update_biases_adam", "adam_v_biases");
    this->core->template addArgument<ulong>("update_biases_adam", numSamples);
    this->core->template addArgument<float>("update_biases_adam", this->trainingConfig.learningRate);
    this->core->template addArgument<float>("update_biases_adam", static_cast<float>(opt.beta1));
    this->core->template addArgument<float>("update_biases_adam", static_cast<float>(opt.beta2));
    this->core->template addArgument<float>("update_biases_adam", static_cast<float>(opt.epsilon));
    this->core->template addArgument<float>("update_biases_adam", bc1);
    this->core->template addArgument<float>("update_biases_adam", bc2);
    this->core->template addArgument<ulong>("update_biases_adam", numBiases);

    this->core->addKernel("update_weights_adam", numWeights, 0);
    this->core->template addArgument<T>("update_weights_adam", "weights");
    this->core->template addArgument<T>("update_weights_adam", "accum_dCost_dWeights");
    this->core->template addArgument<T>("update_weights_adam", "adam_m_weights");
    this->core->template addArgument<T>("update_weights_adam", "adam_v_weights");
    this->core->template addArgument<ulong>("update_weights_adam", numSamples);
    this->core->template addArgument<float>("update_weights_adam", this->trainingConfig.learningRate);
    this->core->template addArgument<float>("update_weights_adam", static_cast<float>(opt.beta1));
    this->core->template addArgument<float>("update_weights_adam", static_cast<float>(opt.beta2));
    this->core->template addArgument<float>("update_weights_adam", static_cast<float>(opt.epsilon));
    this->core->template addArgument<float>("update_weights_adam", bc1);
    this->core->template addArgument<float>("update_weights_adam", bc2);
    this->core->template addArgument<ulong>("update_weights_adam", numWeights);
  } else {
    // SGD
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
}

//===================================================================================================================//
//-- addPropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addPropagateKernels()
{
  ulong numLayers = this->layersConfig.size();

  for (ulong l = 1; l < numLayers; l++) {
    const Layer& layer = this->layersConfig[l];
    ulong numNeurons = layer.numNeurons;
    ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

    // Precompute offsets
    ulong actvOffset = this->bufferManager.getActvOffset(l);
    ulong prevActvOffset = this->bufferManager.getActvOffset(l - 1);
    ulong weightOffset = this->bufferManager.getWeightOffset(l);
    ulong biasOffset = this->bufferManager.getBiasOffset(l);

    std::string calculate_zs_id = "calculate_zs_layer" + std::to_string(l);
    std::string calculate_actvs_id = "calculate_actvs_layer" + std::to_string(l);

    // calculate_zs kernel: parallel reduction — one work-group per neuron
    ulong localWorkSize = 256;

    if (prevNumNeurons <= 64)
      localWorkSize = 32;
    else if (prevNumNeurons <= 256)
      localWorkSize = 64;
    else if (prevNumNeurons <= 1024)
      localWorkSize = 128;

    ulong globalWorkSize = numNeurons * localWorkSize;
    this->core->addKernel(calculate_zs_id, "calculate_zs", globalWorkSize, 0, localWorkSize);
    this->core->template addArgument<T>(calculate_zs_id, "zs");
    this->core->template addArgument<T>(calculate_zs_id, "weights");
    this->core->template addArgument<T>(calculate_zs_id, "actvs");
    this->core->template addArgument<T>(calculate_zs_id, "biases");
    this->core->template addArgument<ulong>(calculate_zs_id, prevNumNeurons);
    this->core->template addArgument<ulong>(calculate_zs_id, weightOffset);
    this->core->template addArgument<ulong>(calculate_zs_id, prevActvOffset);
    this->core->template addArgument<ulong>(calculate_zs_id, biasOffset);
    this->core->template addArgument<ulong>(calculate_zs_id, actvOffset);

    // calculate_actvs kernel
    ulong actvWorkItems = (layer.actvFuncType == ActvFuncType::SOFTMAX) ? 1 : numNeurons;
    this->core->addKernel(calculate_actvs_id, "calculate_actvs", actvWorkItems, 0);

    this->core->template addArgument<T>(calculate_actvs_id, "actvs");
    this->core->template addArgument<T>(calculate_actvs_id, "zs");
    this->core->template addArgument<ulong>(calculate_actvs_id, numNeurons);
    this->core->template addArgument<ulong>(calculate_actvs_id, static_cast<ulong>(layer.actvFuncType));
    this->core->template addArgument<ulong>(calculate_actvs_id, actvOffset);

    // Dropout kernel: apply pre-generated mask after activation (skip last layer)
    if (this->bufferManager.hasDropout && l < numLayers - 1) {
      std::string dropout_id = "apply_dropout_layer" + std::to_string(l);
      this->core->addKernel(dropout_id, "apply_dropout", numNeurons, 0);
      this->core->template addArgument<T>(dropout_id, "actvs");
      this->core->template addArgument<T>(dropout_id, "dropoutMask");
      this->core->template addArgument<ulong>(dropout_id, actvOffset);
    }
  }
}

//===================================================================================================================//
//-- addBackpropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBackpropagateKernels(bool includeInputGradients)
{
  ulong numLayers = this->layersConfig.size();

  // Last layer kernels
  ulong l = numLayers - 1;
  const Layer& lastLayer = this->layersConfig[l];

  ulong numNeurons = lastLayer.numNeurons;
  ulong numBiases = numNeurons;
  ulong numWeights = Utils<T>::count(this->parameters.weights[l]);
  ulong prevNumNeurons = this->layersConfig[l - 1].numNeurons;

  ulong actvOffset = this->bufferManager.getActvOffset(l);
  ulong prevActvOffset = this->bufferManager.getActvOffset(l - 1);
  ulong weightOffset = this->bufferManager.getWeightOffset(l);
  ulong biasOffset = this->bufferManager.getBiasOffset(l);

  // calculate_dCost_dActv_last_layer
  this->core->addKernel("calculate_dCost_dActv_last_layer", numNeurons, 0);
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "dCost_dActvs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "actvs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "outputs");
  this->core->template addArgument<T>("calculate_dCost_dActv_last_layer", "lossWeights");
  this->core->template addArgument<ulong>("calculate_dCost_dActv_last_layer", actvOffset);
  this->core->template addArgument<ulong>("calculate_dCost_dActv_last_layer",
                                          static_cast<ulong>(this->costFunctionConfig.type));

  std::string dCost_dBias_last_id = "calculate_dCost_dBias_layer" + std::to_string(l);
  std::string dCost_dWeight_last_id = "calculate_dCost_dWeight_layer" + std::to_string(l);

  // calculate_dCost_dBias for last layer
  this->core->addKernel(dCost_dBias_last_id, "calculate_dCost_dBias", numBiases, 0);
  this->core->template addArgument<T>(dCost_dBias_last_id, "dCost_dBiases");
  this->core->template addArgument<T>(dCost_dBias_last_id, "actvs");
  this->core->template addArgument<T>(dCost_dBias_last_id, "zs");
  this->core->template addArgument<T>(dCost_dBias_last_id, "dCost_dActvs");
  this->core->template addArgument<ulong>(dCost_dBias_last_id, numNeurons);
  this->core->template addArgument<ulong>(dCost_dBias_last_id, static_cast<ulong>(lastLayer.actvFuncType));
  this->core->template addArgument<ulong>(dCost_dBias_last_id, actvOffset);
  this->core->template addArgument<ulong>(dCost_dBias_last_id, biasOffset);

  // calculate_dCost_dWeight for last layer
  this->core->addKernel(dCost_dWeight_last_id, "calculate_dCost_dWeight", numWeights, 0);
  this->core->template addArgument<T>(dCost_dWeight_last_id, "dCost_dWeights");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "actvs");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "zs");
  this->core->template addArgument<T>(dCost_dWeight_last_id, "dCost_dActvs");
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, prevNumNeurons);
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, numNeurons);
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, static_cast<ulong>(lastLayer.actvFuncType));
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, prevActvOffset);
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, actvOffset);
  this->core->template addArgument<ulong>(dCost_dWeight_last_id, weightOffset);

  // Hidden layers (from second-to-last to first hidden layer)
  for (ulong layer_idx = numLayers - 2; layer_idx >= 1; layer_idx--) {
    const Layer& curr_layer = this->layersConfig[layer_idx];
    const Layer& next_layer = this->layersConfig[layer_idx + 1];

    ulong curr_numNeurons = curr_layer.numNeurons;
    ulong curr_numBiases = curr_numNeurons;
    ulong curr_numWeights = Utils<T>::count(this->parameters.weights[layer_idx]);
    ulong curr_prevNumNeurons = this->layersConfig[layer_idx - 1].numNeurons;
    ulong next_numNeurons = next_layer.numNeurons;

    ulong curr_actvOffset = this->bufferManager.getActvOffset(layer_idx);
    ulong curr_prevActvOffset = this->bufferManager.getActvOffset(layer_idx - 1);
    ulong next_actvOffset = this->bufferManager.getActvOffset(layer_idx + 1);
    ulong next_weightOffset = this->bufferManager.getWeightOffset(layer_idx + 1);
    ulong curr_weightOffset = this->bufferManager.getWeightOffset(layer_idx);
    ulong curr_biasOffset = this->bufferManager.getBiasOffset(layer_idx);

    std::string dCost_dActv_id = "calculate_dCost_dActv_layer" + std::to_string(layer_idx);
    std::string dCost_dBias_id = "calculate_dCost_dBias_layer" + std::to_string(layer_idx);
    std::string dCost_dWeight_id = "calculate_dCost_dWeight_layer" + std::to_string(layer_idx);

    // calculate_dCost_dActv: propagate gradients from next layer
    this->core->addKernel(dCost_dActv_id, "calculate_dCost_dActv", curr_numNeurons, 0);
    this->core->template addArgument<T>(dCost_dActv_id, "dCost_dActvs");
    this->core->template addArgument<T>(dCost_dActv_id, "actvs");
    this->core->template addArgument<T>(dCost_dActv_id, "weights");
    this->core->template addArgument<T>(dCost_dActv_id, "zs");
    this->core->template addArgument<ulong>(dCost_dActv_id, next_numNeurons);
    this->core->template addArgument<ulong>(dCost_dActv_id, curr_numNeurons);
    this->core->template addArgument<ulong>(dCost_dActv_id, static_cast<ulong>(next_layer.actvFuncType));
    this->core->template addArgument<ulong>(dCost_dActv_id, next_weightOffset);
    this->core->template addArgument<ulong>(dCost_dActv_id, next_actvOffset);
    this->core->template addArgument<ulong>(dCost_dActv_id, curr_actvOffset);

    // Apply dropout mask to gradients (same mask as forward pass)
    if (this->bufferManager.hasDropout) {
      std::string dropout_bwd_id = "apply_dropout_backward_layer" + std::to_string(layer_idx);
      this->core->addKernel(dropout_bwd_id, "apply_dropout_backward", curr_numNeurons, 0);
      this->core->template addArgument<T>(dropout_bwd_id, "dCost_dActvs");
      this->core->template addArgument<T>(dropout_bwd_id, "dropoutMask");
      this->core->template addArgument<ulong>(dropout_bwd_id, curr_actvOffset);
    }

    // calculate_dCost_dBias
    this->core->addKernel(dCost_dBias_id, "calculate_dCost_dBias", curr_numBiases, 0);
    this->core->template addArgument<T>(dCost_dBias_id, "dCost_dBiases");
    this->core->template addArgument<T>(dCost_dBias_id, "actvs");
    this->core->template addArgument<T>(dCost_dBias_id, "zs");
    this->core->template addArgument<T>(dCost_dBias_id, "dCost_dActvs");
    this->core->template addArgument<ulong>(dCost_dBias_id, curr_numNeurons);
    this->core->template addArgument<ulong>(dCost_dBias_id, static_cast<ulong>(curr_layer.actvFuncType));
    this->core->template addArgument<ulong>(dCost_dBias_id, curr_actvOffset);
    this->core->template addArgument<ulong>(dCost_dBias_id, curr_biasOffset);

    // calculate_dCost_dWeight
    this->core->addKernel(dCost_dWeight_id, "calculate_dCost_dWeight", curr_numWeights, 0);
    this->core->template addArgument<T>(dCost_dWeight_id, "dCost_dWeights");
    this->core->template addArgument<T>(dCost_dWeight_id, "actvs");
    this->core->template addArgument<T>(dCost_dWeight_id, "zs");
    this->core->template addArgument<T>(dCost_dWeight_id, "dCost_dActvs");
    this->core->template addArgument<ulong>(dCost_dWeight_id, curr_prevNumNeurons);
    this->core->template addArgument<ulong>(dCost_dWeight_id, curr_numNeurons);
    this->core->template addArgument<ulong>(dCost_dWeight_id, static_cast<ulong>(curr_layer.actvFuncType));
    this->core->template addArgument<ulong>(dCost_dWeight_id, curr_prevActvOffset);
    this->core->template addArgument<ulong>(dCost_dWeight_id, curr_actvOffset);
    this->core->template addArgument<ulong>(dCost_dWeight_id, curr_weightOffset);

    // Break condition for ulong loop (can't go negative)
    if (layer_idx == 1)
      break;
  }

  // Optionally compute input layer gradients (layer 0) for external orchestrators (e.g., CNN)
  if (includeInputGradients) {
    ulong inputNumNeurons = this->layersConfig[0].numNeurons;
    ulong layer1NumNeurons = this->layersConfig[1].numNeurons;
    ulong layer1WeightOffset = this->bufferManager.getWeightOffset(1);
    ulong layer1ActvOffset = this->bufferManager.getActvOffset(1);
    ulong layer0ActvOffset = this->bufferManager.getActvOffset(0);

    this->core->addKernel("calculate_dCost_dActv_layer0", "calculate_dCost_dActv", inputNumNeurons, 0);
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "dCost_dActvs");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "actvs");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "weights");
    this->core->template addArgument<T>("calculate_dCost_dActv_layer0", "zs");
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", layer1NumNeurons);
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", inputNumNeurons);
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0",
                                            static_cast<ulong>(this->layersConfig[1].actvFuncType));
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", layer1WeightOffset);
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", layer1ActvOffset);
    this->core->template addArgument<ulong>("calculate_dCost_dActv_layer0", layer0ActvOffset);
  }
}

//===================================================================================================================//
//-- addAccumulateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addAccumulateKernels()
{
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
//-- invalidateAllKernelFlags --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::invalidateAllKernelFlags()
{
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->backpropagateKernelsSetup = false;
  this->accumulateKernelsSetup = false;
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class ANN::GPUKernelBuilder<int>;
template class ANN::GPUKernelBuilder<float>;
template class ANN::GPUKernelBuilder<double>;