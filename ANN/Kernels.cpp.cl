#ifndef KERNELS_CPP_CL
#define KERNELS_CPP_CL

// Note: The following files must be loaded before this file (in order):
//   1. Defines.hpp.cl
//   2. ActvFunc.cpp.cl
//   3. IdxHelper.cpp.cl

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

  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;

  // Calculate offset for this layer in the flattened arrays
  ulong layer2dOffset = 0;
  ulong prevLayer2dOffset = 0;
  ulong layer3dOffset = 0;

  for (ulong ll = 0; ll < layerIdx; ll++) {
    prevLayer2dOffset = layer2dOffset;
    layer2dOffset += layers[ll].numNeurons;
  }

  ulong prevWeightNeurons = layers[0].numNeurons;
  for (ulong ll = 1; ll < layerIdx; ll++) {
    layer3dOffset += layers[ll].numNeurons * prevWeightNeurons;
    prevWeightNeurons = layers[ll].numNeurons;
  }

  TYPE sum = biases[layer2dOffset + j];

  for (ulong k = 0; k < prevNumNeurons; k++) {
    ulong weightIdx = layer3dOffset + j * prevNumNeurons + k;
    ulong prevActvIdx = prevLayer2dOffset + k;

    sum += weights[weightIdx] * actvs[prevActvIdx];
  }

  zs[layer2dOffset + j] = sum;
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

  // Calculate offset for this layer
  ulong layer2dOffset = 0;
  for (ulong ll = 0; ll < layerIdx; ll++) {
    layer2dOffset += layers[ll].numNeurons;
  }

  ulong idx = layer2dOffset + j;
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

  // Calculate offset for the last layer
  ulong lastLayerOffset = 0;
  for (ulong ll = 0; ll < numLayers - 1; ll++) {
    lastLayerOffset += layers[ll].numNeurons;
  }

  ulong idx = lastLayerOffset + j;
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

  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong nextNumNeurons = layers[layerIdx + 1].numNeurons;

  // Calculate offsets
  ulong layer2dOffset = 0;
  for (ulong ll = 0; ll < layerIdx; ll++) {
    layer2dOffset += layers[ll].numNeurons;
  }
  ulong nextLayer2dOffset = layer2dOffset + numNeurons;

  ulong layer3dOffset = 0;
  ulong prevWeightNeurons = layers[0].numNeurons;
  for (ulong ll = 1; ll <= layerIdx; ll++) {
    layer3dOffset += layers[ll].numNeurons * prevWeightNeurons;
    prevWeightNeurons = layers[ll].numNeurons;
  }

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong nextLayer2dIdx = nextLayer2dOffset + j;
    ulong weightIdx = layer3dOffset + j * numNeurons + k;

    TYPE weight = weights[weightIdx];
    TYPE z = zs[nextLayer2dIdx];

    ActvFuncType actvFuncType = layers[layerIdx + 1].actvFuncType;
    TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

    TYPE dCost_dActv_next = dCost_dActvs[nextLayer2dIdx];

    sum += weight * dActvFunc_z * dCost_dActv_next;
  }

  dCost_dActvs[layer2dOffset + k] = sum;
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

  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;

  ulong j = gid / prevNumNeurons;
  ulong k = gid % prevNumNeurons;

  // Calculate offsets
  ulong layer2dOffset = 0;
  ulong prevLayer2dOffset = 0;
  for (ulong ll = 0; ll < layerIdx; ll++) {
    prevLayer2dOffset = layer2dOffset;
    layer2dOffset += layers[ll].numNeurons;
  }

  ulong layer3dOffset = 0;
  ulong prevWeightNeurons = layers[0].numNeurons;
  for (ulong ll = 1; ll < layerIdx; ll++) {
    layer3dOffset += layers[ll].numNeurons * prevWeightNeurons;
    prevWeightNeurons = layers[ll].numNeurons;
  }

  TYPE actv = actvs[prevLayer2dOffset + k];
  TYPE z = zs[layer2dOffset + j];

  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;
  TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

  TYPE dCost_dActv = dCost_dActvs[layer2dOffset + j];

  dCost_dWeights[layer3dOffset + gid] = actv * dActvFunc_z * dCost_dActv;
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

  // Calculate offset for this layer
  ulong layer2dOffset = 0;
  for (ulong ll = 0; ll < layerIdx; ll++) {
    layer2dOffset += layers[ll].numNeurons;
  }

  ulong idx = layer2dOffset + j;
  TYPE z = zs[idx];

  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;
  TYPE dActvFunc_z = actvFunc_derivative(z, actvFuncType);

  TYPE dCost_dActv = dCost_dActvs[idx];

  dCost_dBiases[idx] = dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

#endif // KERNELS_CPP_CL
