#ifndef ANN_BACKPROPAGATE_CPP_CL
#define ANN_BACKPROPAGATE_CPP_CL

// Note: Depends on ANN_Defines.hpp.cl, ANN_ActvFunc.cpp.cl

//===================================================================================================================//

kernel void calculate_dCost_dActv_last_layer(global TYPE* dCost_dActvs, global TYPE* actvs, global TYPE* outputs,
                                             global TYPE* lossWeights, ulong actvOffset, ulong costFunctionType)
{
  size_t j = get_global_id(0);

  ulong idx = actvOffset + j;

  if (costFunctionType == 2) {
    // Cross-entropy: dL/da_j = -w_j * y_j / max(a_j, epsilon)
    TYPE epsilon = (TYPE)1e-7;
    TYPE pred = max(actvs[idx], epsilon);
    dCost_dActvs[idx] = -lossWeights[j] * outputs[j] / pred;
  } else {
    // MSE: dL/da_j = 2 * w_j * (a_j - y_j)
    dCost_dActvs[idx] = 2.0f * lossWeights[j] * (actvs[idx] - outputs[j]);
  }
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(global TYPE* dCost_dActvs, global TYPE* actvs, global TYPE* weights, global TYPE* zs,
                                  ulong nextNumNeurons, ulong prevNumNeurons, ulong actvFuncType, ulong weightOffset,
                                  ulong nextActvOffset, ulong curActvOffset)
{
  size_t k = get_global_id(0);

  TYPE sum = 0.0f;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong weightIdx = weightOffset + j * prevNumNeurons + k;

    TYPE weight = weights[weightIdx];
    TYPE dCost_dZ =
      actvFunc_derivative(actvs, zs, dCost_dActvs, j, nextNumNeurons, (ActvFuncType)actvFuncType, nextActvOffset);

    sum += weight * dCost_dZ;
  }

  dCost_dActvs[curActvOffset + k] = sum;
}

//===================================================================================================================//

kernel void calculate_dCost_dWeight(global TYPE* dCost_dWeights, global TYPE* actvs, global TYPE* zs,
                                    global TYPE* dCost_dActvs, ulong prevNumNeurons, ulong numNeurons,
                                    ulong actvFuncType, ulong prevActvOffset, ulong actvOffset, ulong weightOffset)
{
  size_t gid = get_global_id(0);

  ulong j = gid / prevNumNeurons;
  ulong k = gid % prevNumNeurons;

  TYPE actv = actvs[prevActvOffset + k];
  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, (ActvFuncType)actvFuncType, actvOffset);

  dCost_dWeights[weightOffset + gid] = actv * dCost_dZ;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(global TYPE* dCost_dBiases, global TYPE* actvs, global TYPE* zs,
                                  global TYPE* dCost_dActvs, ulong numNeurons, ulong actvFuncType, ulong actvOffset,
                                  ulong biasOffset)
{
  size_t j = get_global_id(0);

  TYPE dCost_dZ = actvFunc_derivative(actvs, zs, dCost_dActvs, j, numNeurons, (ActvFuncType)actvFuncType, actvOffset);

  dCost_dBiases[biasOffset + j] = dCost_dZ;
}

//===================================================================================================================//

// Applies a pre-generated dropout mask to activations (forward pass, training only).
// mask values are 0 (dropped) or 1/(1-p) (kept with inverted scaling).
// One work-item per neuron in the layer.
kernel void apply_dropout(global TYPE* actvs, global TYPE* dropoutMask, ulong actvOffset)
{
  size_t gid = get_global_id(0);
  actvs[actvOffset + gid] *= dropoutMask[actvOffset + gid];
}

//===================================================================================================================//

// Applies the same dropout mask to gradients during backpropagation.
// Dropped neurons (mask=0) get zero gradient; kept neurons scale by 1/(1-p).
kernel void apply_dropout_backward(global TYPE* dCost_dActvs, global TYPE* dropoutMask, ulong actvOffset)
{
  size_t gid = get_global_id(0);
  dCost_dActvs[actvOffset + gid] *= dropoutMask[actvOffset + gid];
}

//===================================================================================================================//

#endif // ANN_BACKPROPAGATE_CPP_CL
