#ifndef IDXHELPER_CPP_CL
#define IDXHELPER_CPP_CL

#include "../Defines.hpp.cl"

//===================================================================================================================//

void getLJfrom2dIdx(long* l, long* j, long idx, LayersConfig* layersConfig, ulong numLayers) {
  ulong iidx = 0, ll = 0;

  for (ll = 0; ll < numLayers; ll++) {
    if (idx < iidx + layersConfig[ll].numNeurons) {
      *l = ll;
      *j = idx - iidx;
      return;
    } else {
      iidx += layersConfig[ll].numNeurons;
    }
  }
}

//===================================================================================================================//

void getLJKfrom3dIdx(long* l, long* j, long* k, long idx, LayersConfig* layersConfig, ulong numLayers) {
  ulong iidx = 0, ll = 0, jj = 0;

  ulong prevNumNeurons = layersConfig[0].numNeurons;

  for (ll = 1; ll < numLayers; ll++) {
    ulong llNumNeurons = (idx - iidx) / prevNumNeurons;

    if (idx < iidx + layersConfig[ll].numNeurons * prevNumNeurons) {
      *l = ll;
      *j = llNumNeurons;
      *k = idx - (iidx + llNumNeurons * prevNumNeurons);
    } else {
      iidx += layersConfig[ll].numNeurons * prevNumNeurons;
    }

    prevNumNeurons = layersConfig[ll].numNeurons;
  }
}

//===================================================================================================================//

ulong get2dIdxFromLJ(long l, long j, LayersConfig* layersConfig) {
  ulong idx = 0, ll = 0;

  for (ll = 0; ll < l; ll++) {
    idx += layersConfig[ll].numNeurons;
  }

  return idx + j;
}

//===================================================================================================================//

ulong get3dIdxFromLJL(long l, long j, long k, LayersConfig* layersConfig) {
  ulong idx = 0, ll = 0;

  ulong prevNumNeurons = layersConfig[0].numNeurons;

  for (ll = 1; ll < l; ll++) {
    idx += layersConfig[ll].numNeurons * prevNumNeurons;

    prevNumNeurons = layersConfig[ll].numNeurons;
  }

  return j * prevNumNeurons + k;
}

//===================================================================================================================//
