#ifndef KERNELS_CPP_CL
#define KERNELS_CPP_CL

// Note: Depends on Defines.hpp.cl, ActvFunc.cpp.cl, IdxHelper.cpp.cl
// Files are concatenated by C++ code in load order

//===================================================================================================================//
// Kernels
//===================================================================================================================//

kernel void calculate_zs(
    global TYPE* zs,
    global TYPE* weights,
    global TYPE* actvs,
    global TYPE* biases,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t j = get_global_id(0);

  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;

  TYPE sum = biases[getBiasIndex(layerIdx, j, layers, numLayers)];

  for (ulong k = 0; k < prevNumNeurons; k++) {
    ulong weightIdx = getWeightIndex(layerIdx, j, k, layers, numLayers);
    ulong prevActvIdx = getActvIndex(layerIdx - 1, k, layers, numLayers);

    sum += weights[weightIdx] * actvs[prevActvIdx];
  }

  zs[getZIndex(layerIdx, j, layers, numLayers)] = sum;
}

//===================================================================================================================//

kernel void calculate_actvs(
    global TYPE* actvs,
    global TYPE* zs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  ulong numNeurons = layers[layerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  actvFunc_calculate(zs, actvs, numNeurons, actvFuncType, layerIdx, layers, numLayers);
}

//===================================================================================================================//

kernel void accumulate_dCost_dBiases(
    global TYPE* accum_dCost_dBiases,
    global TYPE* dCost_dBiases,
    ulong size
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    accum_dCost_dBiases[idx] += dCost_dBiases[idx];
  }
}

//===================================================================================================================//

kernel void accumulate_dCost_dWeights(
    global TYPE* accum_dCost_dWeights,
    global TYPE* dCost_dWeights,
    ulong size
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    accum_dCost_dWeights[idx] += dCost_dWeights[idx];
  }
}

//===================================================================================================================//

kernel void update_biases(
    global TYPE* biases,
    global TYPE* accum_dCost_dBiases,
    ulong numSamples,
    float learningRate,
    ulong size
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    biases[idx] -= learningRate * (accum_dCost_dBiases[idx] / (TYPE)numSamples);
  }
}

//===================================================================================================================//

kernel void update_weights(
    global TYPE* weights,
    global TYPE* accum_dCost_dWeights,
    ulong numSamples,
    float learningRate,
    ulong size
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    weights[idx] -= learningRate * (accum_dCost_dWeights[idx] / (TYPE)numSamples);
  }
}

//===================================================================================================================//

kernel void calculate_dCost_dActv_last_layer(
    global TYPE* dCost_dActvs,
    global TYPE* actvs,
    global TYPE* outputs,
    global TYPE* lossWeights,
    ulong numOutputNeurons,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t j = get_global_id(0);

  ulong lastLayerIdx = numLayers - 1;
  ulong idx = getActvIndex(lastLayerIdx, j, layers, numLayers);

  dCost_dActvs[idx] = 2.0f * lossWeights[j] * (actvs[idx] - outputs[j]);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(
    global TYPE* dCost_dActvs,
    global TYPE* actvs,
    global TYPE* weights,
    global TYPE* zs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t k = get_global_id(0);

  ulong nextLayerIdx = layerIdx + 1;
  ulong nextNumNeurons = layers[nextLayerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[nextLayerIdx].actvFuncType;

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong weightIdx = getWeightIndex(nextLayerIdx, j, k, layers, numLayers);

    TYPE weight = weights[weightIdx];
    TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, nextNumNeurons, actvFuncType,
                                        nextLayerIdx, layers, numLayers);

    sum += weight * dCost_dZ;
  }

  dCost_dActvs[getActvIndex(layerIdx, k, layers, numLayers)] = sum;
}

//===================================================================================================================//

kernel void calculate_dCost_dWeight(
    global TYPE* dCost_dWeights,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t gid = get_global_id(0);

  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;
  ulong numNeurons = layers[layerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  ulong j = gid / prevNumNeurons;
  ulong k = gid % prevNumNeurons;

  TYPE actv = actvs[getActvIndex(layerIdx - 1, k, layers, numLayers)];
  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, actvFuncType,
                                      layerIdx, layers, numLayers);

  dCost_dWeights[getWeightIndex(layerIdx, j, k, layers, numLayers)] = actv * dCost_dZ;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(
    global TYPE* dCost_dBiases,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t j = get_global_id(0);

  ulong numNeurons = layers[layerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, actvFuncType,
                                      layerIdx, layers, numLayers);

  dCost_dBiases[getBiasIndex(layerIdx, j, layers, numLayers)] = dCost_dZ;
}

//===================================================================================================================//

#endif // KERNELS_CPP_CL
