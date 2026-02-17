#ifndef IDXHELPER_CPP_CL
#define IDXHELPER_CPP_CL

#include "Defines.hpp.cl"

//===================================================================================================================//

ulong getZIndex(ulong l, ulong j, constant Layer* layers, ulong numLayers) {
  ulong idx = 0;
  for (ulong i = 0; i < l; i++) {
    idx += layers[i].numNeurons;
  }
  return idx + j;
}

//===================================================================================================================//

ulong getActvIndex(ulong l, ulong j, constant Layer* layers, ulong numLayers) {
  return getZIndex(l, j, layers, numLayers);
}

//===================================================================================================================//

ulong getWeightIndex(ulong l, ulong j, ulong k, constant Layer* layers, ulong numLayers) {
  ulong idx = 0;
  for (ulong i = 1; i < l; i++) {
    idx += layers[i].numNeurons * layers[i - 1].numNeurons;
  }
  return idx + j * layers[l - 1].numNeurons + k;
}

//===================================================================================================================//

ulong getBiasIndex(ulong l, ulong j, constant Layer* layers, ulong numLayers) {
  ulong idx = 0;
  for (ulong i = 1; i < l; i++) {
    idx += layers[i].numNeurons;
  }
  return idx + j;
}

//===================================================================================================================//

#endif // IDXHELPER_CPP_CL
