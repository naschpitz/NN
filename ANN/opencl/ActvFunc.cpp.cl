#ifndef ACTVFUNC_CPP_CL
#define ACTVFUNC_CPP_CL

// Note: Depends on Defines.hpp.cl (TYPE, ActvFuncType)
// Files are concatenated by C++ code in load order

//===================================================================================================================//

TYPE actvFunc_relu(TYPE x) {
  return (x > 0) ? x : 0;
}

//===================================================================================================================//

TYPE actvFunc_sigmoid(TYPE x) {
  return 1.0f / (1.0f + exp(-x));
}

//===================================================================================================================//

TYPE actvFunc_tanh_impl(TYPE x) {
  return tanh(x);
}

//===================================================================================================================//

void actvFunc_softmax(global TYPE* zs, global TYPE* actvs, ulong numNeurons,
                      ulong layerIdx, constant Layer* layers, ulong numLayers,
                      ulong batchOffset) {
  // Find max z for numerical stability
  TYPE maxZ = zs[batchOffset + getZIndex(layerIdx, 0, layers, numLayers)];
  for (ulong j = 1; j < numNeurons; j++) {
    TYPE z = zs[batchOffset + getZIndex(layerIdx, j, layers, numLayers)];
    if (z > maxZ) maxZ = z;
  }

  // Compute exp(z - maxZ) and sum
  TYPE sumExp = 0.0f;
  for (ulong j = 0; j < numNeurons; j++) {
    ulong idx = batchOffset + getActvIndex(layerIdx, j, layers, numLayers);
    TYPE z = zs[batchOffset + getZIndex(layerIdx, j, layers, numLayers)];
    actvs[idx] = exp(z - maxZ);
    sumExp += actvs[idx];
  }

  // Normalize
  for (ulong j = 0; j < numNeurons; j++) {
    ulong idx = batchOffset + getActvIndex(layerIdx, j, layers, numLayers);
    actvs[idx] /= sumExp;
  }
}

//===================================================================================================================//

TYPE actvFunc_drelu(TYPE x) {
  return (x > 0) ? 1.0f : 0.0f;
}

//===================================================================================================================//

TYPE actvFunc_dsigmoid(TYPE x) {
  TYPE sig = actvFunc_sigmoid(x);
  return sig * (1.0f - sig);
}

//===================================================================================================================//

TYPE actvFunc_dtanh(TYPE x) {
  TYPE t = tanh(x);
  return 1.0f - (t * t);
}

//===================================================================================================================//

// Softmax derivative for neuron j: dCost/dZ_j = s_j * (dCost/dActv_j - dot)
// where dot = Σ s_i * dCost/dActv_i
TYPE actvFunc_dsoftmax(global TYPE* actvs, global TYPE* dCost_dActvs, ulong j, ulong numNeurons,
                       ulong layerIdx, constant Layer* layers, ulong numLayers,
                       ulong batchOffset) {
  TYPE dot = 0.0f;
  for (ulong i = 0; i < numNeurons; i++) {
    ulong idx = batchOffset + getActvIndex(layerIdx, i, layers, numLayers);
    dot += actvs[idx] * dCost_dActvs[idx];
  }

  ulong actvIdx = batchOffset + getActvIndex(layerIdx, j, layers, numLayers);
  return actvs[actvIdx] * (dCost_dActvs[actvIdx] - dot);
}

//===================================================================================================================//

// Layer-wide forward activation.
// For element-wise types (relu, sigmoid, tanh): each work-item computes actvs[j] from zs[j].
// For softmax: single work-item computes all actvs from all zs.
// j and batchOffset are computed by the caller (kernel), not via get_global_id.
void actvFunc_calculate(global TYPE* zs, global TYPE* actvs, ulong numNeurons, ActvFuncType type,
                        ulong layerIdx, constant Layer* layers, ulong numLayers,
                        ulong j, ulong batchOffset) {
  ulong idx = batchOffset + getZIndex(layerIdx, j, layers, numLayers);
  TYPE z = zs[idx];

  switch(type) {
    case ACTV_RELU:
      actvs[idx] = actvFunc_relu(z);
      break;
    case ACTV_SIGMOID:
      actvs[idx] = actvFunc_sigmoid(z);
      break;
    case ACTV_TANH:
      actvs[idx] = actvFunc_tanh_impl(z);
      break;
    case ACTV_SOFTMAX:
      actvFunc_softmax(zs, actvs, numNeurons, layerIdx, layers, numLayers, batchOffset);
      break;
    default:
      actvs[idx] = z;
      break;
  }
}

//===================================================================================================================//

// Layer-wide derivative: computes dCost/dZ for neuron j.
TYPE actvFunc_derivative(global TYPE* actvs, global TYPE* zs, global TYPE* dCost_dActvs,
                         ulong j, ulong numNeurons, ActvFuncType type,
                         ulong layerIdx, constant Layer* layers, ulong numLayers,
                         ulong batchOffset) {
  ulong zIdx = batchOffset + getZIndex(layerIdx, j, layers, numLayers);
  TYPE z = zs[zIdx];
  TYPE dCost_dActv = dCost_dActvs[batchOffset + getActvIndex(layerIdx, j, layers, numLayers)];

  TYPE dActvFunc_z;

  switch(type) {
    case ACTV_RELU:
      dActvFunc_z = actvFunc_drelu(z);
      break;
    case ACTV_SIGMOID:
      dActvFunc_z = actvFunc_dsigmoid(z);
      break;
    case ACTV_TANH:
      dActvFunc_z = actvFunc_dtanh(z);
      break;
    case ACTV_SOFTMAX:
      return actvFunc_dsoftmax(actvs, dCost_dActvs, j, numNeurons, layerIdx, layers, numLayers, batchOffset);
    default:
      dActvFunc_z = 1.0f;
      break;
  }

  return dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

#endif // ACTVFUNC_CPP_CL
