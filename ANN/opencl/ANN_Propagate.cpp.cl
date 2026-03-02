#ifndef ANN_PROPAGATE_CPP_CL
#define ANN_PROPAGATE_CPP_CL

// Note: Depends on ANN_Defines.hpp.cl, ANN_ActvFunc.cpp.cl, ANN_IdxHelper.cpp.cl

//===================================================================================================================//

// calculate_zs: computes z = W*a + b for each neuron using parallel reduction.
// Each neuron gets one work-group of localWorkSize work-items.
// Global work size = numNeurons * localWorkSize.
// Each work-item computes a partial dot product, then work-group reduces.
kernel void calculate_zs(global TYPE* zs, global TYPE* weights, global TYPE* actvs, global TYPE* biases,
                         ulong prevNumNeurons, ulong weightOffset, ulong prevActvOffset, ulong biasOffset,
                         ulong zOffset)
{
  local TYPE partials[256];
  size_t groupId = get_group_id(0); // neuron index
  size_t lid = get_local_id(0);
  size_t localSize = get_local_size(0);

  ulong j = groupId; // neuron index
  ulong wRow = weightOffset + j * prevNumNeurons;

  // Each work-item computes partial dot product over its chunk
  TYPE sum = (TYPE)0;

  for (ulong k = lid; k < prevNumNeurons; k += localSize) {
    sum += weights[wRow + k] * actvs[prevActvOffset + k];
  }

  partials[lid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Tree reduction within work-group
  for (ulong stride = localSize / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      partials[lid] += partials[lid + stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Work-item 0 writes the final result
  if (lid == 0) {
    zs[zOffset + j] = partials[0] + biases[biasOffset + j];
  }
}

//===================================================================================================================//

kernel void calculate_actvs(global TYPE* actvs, global TYPE* zs, ulong numNeurons, ulong actvFuncType, ulong actvOffset)
{
  actvFunc_calculate(zs, actvs, numNeurons, (ActvFuncType)actvFuncType, actvOffset);
}

//===================================================================================================================//

#endif // ANN_PROPAGATE_CPP_CL
