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
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong batchIdx = gid / numNeurons;
  ulong j = gid % numNeurons;
  ulong batchOffset = batchIdx * stride;

  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;

  TYPE sum = biases[getBiasIndex(layerIdx, j, layers, numLayers)];

  for (ulong k = 0; k < prevNumNeurons; k++) {
    ulong weightIdx = getWeightIndex(layerIdx, j, k, layers, numLayers);
    ulong prevActvIdx = batchOffset + getActvIndex(layerIdx - 1, k, layers, numLayers);

    sum += weights[weightIdx] * actvs[prevActvIdx];
  }

  zs[batchOffset + getZIndex(layerIdx, j, layers, numLayers)] = sum;
}

//===================================================================================================================//

kernel void calculate_actvs(
    global TYPE* actvs,
    global TYPE* zs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong numNeurons = layers[layerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  // For softmax: work items = batchSize (one per sample), j is unused
  // For element-wise: work items = batchSize * numNeurons
  ulong batchIdx, j;
  if (actvFuncType == ACTV_SOFTMAX) {
    batchIdx = gid;
    j = 0; // unused, softmax iterates internally
  } else {
    batchIdx = gid / numNeurons;
    j = gid % numNeurons;
  }
  ulong batchOffset = batchIdx * stride;

  actvFunc_calculate(zs, actvs, numNeurons, actvFuncType, layerIdx, layers, numLayers, j, batchOffset);
}

//===================================================================================================================//

kernel void accumulate_dCost_dBiases(
    global TYPE* accum_dCost_dBiases,
    global TYPE* dCost_dBiases,
    ulong size,
    ulong batchSize
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    TYPE sum = 0;
    for (ulong b = 0; b < batchSize; b++) {
      sum += dCost_dBiases[b * size + idx];
    }
    accum_dCost_dBiases[idx] += sum;
  }
}

//===================================================================================================================//

kernel void accumulate_dCost_dWeights(
    global TYPE* accum_dCost_dWeights,
    global TYPE* dCost_dWeights,
    ulong size,
    ulong batchSize
  ) {
  size_t idx = get_global_id(0);

  if (idx < size) {
    TYPE sum = 0;
    for (ulong b = 0; b < batchSize; b++) {
      sum += dCost_dWeights[b * size + idx];
    }
    accum_dCost_dWeights[idx] += sum;
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
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong batchIdx = gid / numOutputNeurons;
  ulong j = gid % numOutputNeurons;
  ulong batchOffset = batchIdx * stride;
  ulong outputBatchOffset = batchIdx * numOutputNeurons;

  ulong lastLayerIdx = numLayers - 1;
  ulong idx = batchOffset + getActvIndex(lastLayerIdx, j, layers, numLayers);

  dCost_dActvs[idx] = 2.0f * lossWeights[j] * (actvs[idx] - outputs[outputBatchOffset + j]);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(
    global TYPE* dCost_dActvs,
    global TYPE* actvs,
    global TYPE* weights,
    global TYPE* zs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong batchIdx = gid / numNeurons;
  ulong k = gid % numNeurons;
  ulong batchOffset = batchIdx * stride;

  ulong nextLayerIdx = layerIdx + 1;
  ulong nextNumNeurons = layers[nextLayerIdx].numNeurons;
  ActvFuncType actvFuncType = layers[nextLayerIdx].actvFuncType;

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong weightIdx = getWeightIndex(nextLayerIdx, j, k, layers, numLayers);

    TYPE weight = weights[weightIdx];
    TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, nextNumNeurons, actvFuncType,
                                        nextLayerIdx, layers, numLayers, batchOffset);

    sum += weight * dCost_dZ;
  }

  dCost_dActvs[batchOffset + getActvIndex(layerIdx, k, layers, numLayers)] = sum;
}

//===================================================================================================================//

kernel void calculate_dCost_dWeight(
    global TYPE* dCost_dWeights,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);

  ulong prevNumNeurons = layers[layerIdx - 1].numNeurons;
  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong numWeightsPerLayer = numNeurons * prevNumNeurons;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  ulong batchIdx = gid / numWeightsPerLayer;
  ulong localIdx = gid % numWeightsPerLayer;
  ulong j = localIdx / prevNumNeurons;
  ulong k = localIdx % prevNumNeurons;
  ulong batchOffset = batchIdx * stride;
  ulong weightBatchOffset = batchIdx * numWeightsPerLayer;

  TYPE actv = actvs[batchOffset + getActvIndex(layerIdx - 1, k, layers, numLayers)];
  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, actvFuncType,
                                      layerIdx, layers, numLayers, batchOffset);

  // dCost_dWeights is laid out as [batchSize][numWeightsPerLayer] for this layer's kernel
  // but the buffer is flat across all layers, so we use the layer weight index + batch offset
  ulong totalNumWeights = 0;
  for (ulong i = 1; i < numLayers; i++) {
    totalNumWeights += layers[i].numNeurons * layers[i - 1].numNeurons;
  }
  dCost_dWeights[batchIdx * totalNumWeights + getWeightIndex(layerIdx, j, k, layers, numLayers)] = actv * dCost_dZ;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(
    global TYPE* dCost_dBiases,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong layerIdx,
    constant Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);

  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong batchIdx = gid / numNeurons;
  ulong j = gid % numNeurons;
  ulong batchOffset = batchIdx * stride;
  ActvFuncType actvFuncType = layers[layerIdx].actvFuncType;

  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, actvFuncType,
                                      layerIdx, layers, numLayers, batchOffset);

  // dCost_dBiases is laid out as [batchSize][totalNumBiases]
  ulong totalNumBiases = 0;
  for (ulong i = 1; i < numLayers; i++) {
    totalNumBiases += layers[i].numNeurons;
  }
  dCost_dBiases[batchIdx * totalNumBiases + getBiasIndex(layerIdx, j, layers, numLayers)] = dCost_dZ;
}

//===================================================================================================================//

// Applies a pre-generated dropout mask to activations (forward pass, training only).
// mask values are 0 (dropped) or 1/(1-p) (kept with inverted scaling).
// Work items = batchSize * numNeurons.
kernel void apply_dropout(
    global TYPE* actvs,
    global TYPE* dropoutMask,
    ulong layerIdx,
    global Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong batchIdx = gid / numNeurons;
  ulong localIdx = gid % numNeurons;
  ulong batchOffset = batchIdx * stride;

  // Compute flat activation offset for this layer
  ulong offset = 0;
  for (ulong l = 0; l < layerIdx; l++) offset += layers[l].numNeurons;

  actvs[batchOffset + offset + localIdx] *= dropoutMask[batchOffset + offset + localIdx];
}

//===================================================================================================================//

// Applies the same dropout mask to gradients during backpropagation.
// Dropped neurons (mask=0) get zero gradient; kept neurons scale by 1/(1-p).
kernel void apply_dropout_backward(
    global TYPE* dCost_dActvs,
    global TYPE* dropoutMask,
    ulong layerIdx,
    global Layer* layers,
    ulong numLayers,
    ulong batchSize,
    ulong stride
  ) {
  size_t gid = get_global_id(0);
  ulong numNeurons = layers[layerIdx].numNeurons;
  ulong batchIdx = gid / numNeurons;
  ulong localIdx = gid % numNeurons;
  ulong batchOffset = batchIdx * stride;

  ulong offset = 0;
  for (ulong l = 0; l < layerIdx; l++) offset += layers[l].numNeurons;

  dCost_dActvs[batchOffset + offset + localIdx] *= dropoutMask[batchOffset + offset + localIdx];
}

//===================================================================================================================//

#endif // KERNELS_CPP_CL
