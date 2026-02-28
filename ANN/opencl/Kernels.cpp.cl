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
    ulong prevNumNeurons,
    ulong weightOffset,
    ulong prevActvOffset,
    ulong biasOffset,
    ulong zOffset,
    local TYPE* tile
  ) {
  size_t j = get_global_id(0);
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  TYPE sum = biases[biasOffset + j];

  ulong wRow = weightOffset + j * prevNumNeurons;

  // Process input activations in tiles loaded cooperatively into local memory
  for (ulong tileStart = 0; tileStart < prevNumNeurons; tileStart += localSize) {
    // Cooperatively load a tile of input activations into local memory
    ulong loadIdx = tileStart + lid;
    if (loadIdx < prevNumNeurons)
      tile[lid] = actvs[prevActvOffset + loadIdx];
    else
      tile[lid] = (TYPE)0;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Each work-item multiplies its weights with the shared tile
    ulong tileEnd = tileStart + localSize;
    if (tileEnd > prevNumNeurons) tileEnd = prevNumNeurons;
    ulong tileLen = tileEnd - tileStart;

    for (ulong t = 0; t < tileLen; t++) {
      sum += weights[wRow + tileStart + t] * tile[t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  zs[zOffset + j] = sum;
}

//===================================================================================================================//

kernel void calculate_actvs(
    global TYPE* actvs,
    global TYPE* zs,
    ulong numNeurons,
    ulong actvFuncType,
    ulong actvOffset
  ) {
  actvFunc_calculate(zs, actvs, numNeurons, (ActvFuncType)actvFuncType, actvOffset);
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
    ulong actvOffset
  ) {
  size_t j = get_global_id(0);

  ulong idx = actvOffset + j;

  dCost_dActvs[idx] = 2.0f * lossWeights[j] * (actvs[idx] - outputs[j]);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(
    global TYPE* dCost_dActvs,
    global TYPE* actvs,
    global TYPE* weights,
    global TYPE* zs,
    ulong nextNumNeurons,
    ulong prevNumNeurons,
    ulong actvFuncType,
    ulong weightOffset,
    ulong nextActvOffset,
    ulong curActvOffset
  ) {
  size_t k = get_global_id(0);

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong weightIdx = weightOffset + j * prevNumNeurons + k;

    TYPE weight = weights[weightIdx];
    TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, nextNumNeurons,
                                        (ActvFuncType)actvFuncType, nextActvOffset);

    sum += weight * dCost_dZ;
  }

  dCost_dActvs[curActvOffset + k] = sum;
}

//===================================================================================================================//

kernel void calculate_dCost_dWeight(
    global TYPE* dCost_dWeights,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong prevNumNeurons,
    ulong numNeurons,
    ulong actvFuncType,
    ulong prevActvOffset,
    ulong actvOffset,
    ulong weightOffset
  ) {
  size_t gid = get_global_id(0);

  ulong j = gid / prevNumNeurons;
  ulong k = gid % prevNumNeurons;

  TYPE actv = actvs[prevActvOffset + k];
  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons,
                                      (ActvFuncType)actvFuncType, actvOffset);

  dCost_dWeights[weightOffset + gid] = actv * dCost_dZ;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(
    global TYPE* dCost_dBiases,
    global TYPE* actvs,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong numNeurons,
    ulong actvFuncType,
    ulong actvOffset,
    ulong biasOffset
  ) {
  size_t j = get_global_id(0);

  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons,
                                      (ActvFuncType)actvFuncType, actvOffset);

  dCost_dBiases[biasOffset + j] = dCost_dZ;
}

//===================================================================================================================//

// Applies a pre-generated dropout mask to activations (forward pass, training only).
// mask values are 0 (dropped) or 1/(1-p) (kept with inverted scaling).
// One work-item per neuron in the layer.
kernel void apply_dropout(
    global TYPE* actvs,
    global TYPE* dropoutMask,
    ulong actvOffset
  ) {
  size_t gid = get_global_id(0);
  actvs[actvOffset + gid] *= dropoutMask[actvOffset + gid];
}

//===================================================================================================================//

// Applies the same dropout mask to gradients during backpropagation.
// Dropped neurons (mask=0) get zero gradient; kept neurons scale by 1/(1-p).
kernel void apply_dropout_backward(
    global TYPE* dCost_dActvs,
    global TYPE* dropoutMask,
    ulong actvOffset
  ) {
  size_t gid = get_global_id(0);
  dCost_dActvs[actvOffset + gid] *= dropoutMask[actvOffset + gid];
}

//===================================================================================================================//

#endif // KERNELS_CPP_CL
