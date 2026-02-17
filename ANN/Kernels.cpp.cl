#ifndef KERNELS_CPP_CL
#define KERNELS_CPP_CL

#include "../ActvFunc.cpp.cl"
#include "../Defines.hpp.cl"
#include "../IdxHelper.cpp.cl"

//===================================================================================================================//

kernel void calculate_zs(
  global TYPE* zs,
  global TYPE* weights,
  global TYPE* actvs,
  ulong l,
  constant LayersConfig* layersConfig,
  ulong numLayers
  ) {
  size_t idx = get_global_id(0);

  ulong l, j;
  getLJfrom2dIdx(&l, &j, idx, layersConfig, numLayers);

  ulong prevNumNeurons = layersConfig[l - 1].numNeurons;

  TYPE sum = 0;

  for (ulong k = 0; k < prevNumNeurons; k++) {
    ulong layer3dIdx = get3dIdxFromLJK(l, j, k, layersConfig);
    ulong prevLayer2dIdx = get2dIdxFromLJ(l - 1, k, layersConfig);

    sum += weights[layer3dIdx] * actvs[prevLayer2dIdx];
  }

  zs[idx] = sum;
}

//===================================================================================================================//

kernerl void calculate_actvs(
  global TYPE* actvs,
  global TYPE* zs,
  ulong l,
  constant LayersConfig* layersConfig,
  ulong numLayers
  ) {
  size_t idx = get_global_id(0);

  TYPE z = zs[idx];

  ActvFuncType actvFuncType = layersConfig[l].actvFuncType;
  actvs[idx] = actvFunc_calculate(z, actvFuncType, false);
}

//===================================================================================================================//

kernel void accumulate_dCost_dBiases(
    global TYPE* accum_dCost_dBiases,
    global TYPE* dCost_dBiases
  ) {
  size_t idx = get_global_id(0);

  accum_dCost_dBiases[idx] += accum_dCost_dBiases[idx];
}

//===================================================================================================================//

kernel void accumulate_dCost_dWeights(
    global TYPE* accum_dCost_dWeights,
    global TYPE* dCost_dWeights
  ) {
  size_t idx = get_global_id(0);

  accum_dCost_dWeights[idx] += dCost_dWeights[idx];
}

//===================================================================================================================//

kernel void update_biases(
    global TYPE* biases,
    global TYPE* accum_dCost_dBiases,
    ulong numSamples,
    float learningRate
  ) {
  size_t idx = get_global_id(0);

  biases[idx] -= learningRate * (accum_dCost_dBiases[idx] / numSamples);
}

//===================================================================================================================//

kernel void update_weights(
    global TYPE* weights,
    global TYPE* accum_dCost_dWeights,
    ulong numSamples,
    float learningRate
  ) {
  size_t idx = get_global_id(0);

  weights[idx] -= learningRate * (accum_dCost_dWeights[idx] / numSamples);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv_last_layer(
    global TYPE* dCost_dAcvts,
    global TYPE* acvts,
    global TYPE* outputs,
    ulong numOutputNeurons,
    constant LayersConfig* layersConfig,
    ulong numLayers
  ) {
  size_t idx = get_global_id(0);

  ulong j;
  getLJfrom2dIdx(null, &j, idx, layersConfig, numLayers);
  ulong outputIdx = outputOffset + j;

  return 2 * (this->actvs[idx] - outputs[j]);
}

//===================================================================================================================//

kernel void calculate_dCost_dActv(
    global TYPE* dCost_dAcvts,
    global TYPE* acvts,
    global TYPE* weights,
    global TYPE* zs,
    ulong l,
    constant LayersConfig* layersConfig,
    ulong numLayers
  ) {
  size_t idx = get_global_id(0);

  ulong l, k
  getLJfrom2dIdx(&l, &k, idx, layersConfig, numLayers);

  ulong nextNumNeurons = layersConfig[l + 1].numNeurons;

  TYPE sum = 0;

  for (ulong j = 0; j < nextNumNeurons; j++) {
    ulong nextLayer2dIdx = get2dIdxFromLJ(l + 1, j, layersConfig);
    ulong nextLayer3dIdx = get3dIdxFromLJK(l + 1, j, k, layersConfig);

    TYPE weight = weights[nextLayer3dIdx];
    TYPE z = zs[nextLayer2dIdx];

    ActvFuncType actvFuncType = layersConfig[l + 1].actvFuncType;
    TYPE dActvFunc_z = actvFunc_calculate(z, actvFuncType, true);

    TYPE& dCost_dActv = dCost_dAcvts[nextLayer2dIdx];

    sum += weight * dActvFunc_z * dConst_dActcv;
  }

  dCost_dActv[idx] = sum;
}

//===================================================================================================================//

kernel void calculate_dCost_dWeight(
    global TYPE* dCost_dWeights,
    global TYPE* acvts,
    global TYPE* zs,
    global TYPE* dCost_dAcvts,
    ulong l,
    constant LayersConfig* layersConfig,
    ulong numLayers
  ) {
  size_t idx = get_global_id(0);

  ulong l, j k;
  getLJKfrom3dIdx(&l, &j, &k, idx, layersConfig, numLayers);

  ulong prevLayer2dIdx = get2dIdxFromLJ(l - 1, k, layersConfig);
  TYPE actv = this->actvs[prevLayer2dIdx];

  ulong layer2dIdx = get2dIdxFromLJ(l, j, layersConfig);
  TYPE z = zs[layer2dIdx];

  ActvFuncType actvFuncType = layersConfig[l].actvFuncType;
  TYPE dActvFunc_z = actvFunc_calculate(z, actvFuncType, true);

  TYPE& dCost_dActv = dCost_dAcvts[layer2dIdx];

  dCost_dWeights[idx] = actv * dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

kernel void calculate_dCost_dBias(
    global TYPE* dCost_dBiases,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    ulong l,
    constant LayersConfig* layersConfig
  ) {
  size_t idx = get_global_id(0);

  TYPE& z = zs[idx];

  ActvFuncType actvFuncType = layersConfig[l].actvFuncType;
  TYPE dActvFunc_z = actvFunc_calculate(z, actvFuncType, true);

  TYPE& dCost_dActv = dCost_dActvs[idx];

  dCost_dBiases[idx] = dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//
