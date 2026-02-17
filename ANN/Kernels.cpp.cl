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
  size_t j = get_global_id(0);

  ulong idx = getZIndex(layerIdx, j, layers, numLayers);
  TYPE z = zs[idx];

  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;
  actvs[idx] = actvFunc_calculate(z, actvFuncType);
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
    ulong numOutputNeurons,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t j = get_global_id(0);

  ulong lastLayerIdx = numLayers - 1;
  ulong idx = getActvIndex(lastLayerIdx, j, layers, numLayers);

  dCost_dActvs[idx] = 2.0f * (actvs[idx] - outputs[j]);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(
    global TYPE* dCost_dActvs,
    global TYPE* weights,
    global TYPE* zs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t k = get_global_id(0);

  ulong nextLayerIdx = layerIdx + 1;
  ulong nextNumNeurons = layers[nextLayerIdx].numNeurons;

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong nextZIdx = getZIndex(nextLayerIdx, j, layers, numLayers);
    ulong weightIdx = getWeightIndex(nextLayerIdx, j, k, layers, numLayers);

    TYPE weight = weights[weightIdx];
    TYPE z = zs[nextZIdx];

    ActvFuncType actvFuncType = layers[nextLayerIdx].actvFuncType;
    TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

    TYPE dCost_dActv_next = dCost_dActvs[nextZIdx];

    sum += weight * dActvFunc_z * dCost_dActv_next;
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

  ulong j = gid / prevNumNeurons;
  ulong k = gid % prevNumNeurons;

  TYPE actv = actvs[getActvIndex(layerIdx - 1, k, layers, numLayers)];
  TYPE z = zs[getZIndex(layerIdx, j, layers, numLayers)];

  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;
  TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

  TYPE dCost_dActv = dCost_dActvs[getActvIndex(layerIdx, j, layers, numLayers)];

  dCost_dWeights[getWeightIndex(layerIdx, j, k, layers, numLayers)] = actv * dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(
    global TYPE* dCost_dBiases,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers
  ) {
  size_t j = get_global_id(0);

  ulong zIdx = getZIndex(layerIdx, j, layers, numLayers);
  TYPE z = zs[zIdx];

  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;
  TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

  TYPE dCost_dActv = dCost_dActvs[getActvIndex(layerIdx, j, layers, numLayers)];

  dCost_dBiases[getBiasIndex(layerIdx, j, layers, numLayers)] = dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

#endif // KERNELS_CPP_CL
