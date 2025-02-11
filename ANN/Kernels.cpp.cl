#ifndef KERNELS_CPP_CL
#define KERNELS_CPP_CL

#define TYPE float

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

kernel void update_dCost_dBiases(
    global TYPE* dCost_dBiases,
    global TYPE* accum_dCost_dBiases,
    uint numSamples,
    float learningRate
  ) {
  size_t idx = get_global_id(0);

  dCost_dBiases[idx] = learningRate * (accum_dCost_dBiases[idx] / numSamples);
}

//===================================================================================================================//

kernel void update_dCost_dBWeights(
    global TYPE* dCost_dWeights,
    global TYPE* accum_dCost_dWeights,
    uint numSamples,
    float learningRate
  ) {
  size_t idx = get_global_id(0);

  dCost_dWeights[idx] = learningRate * (accum_dCost_dWeights[idx] / numSamples);
}

//===================================================================================================================//

kernel void calc_dCost_dActv(
    global TYPE* dCost_dAcvts,
    global TYPE* acvts,
    global TYPE* outputs,
    uint numOutputNeurons,
    uint outputOffset,
    constant LayersConfig* layersConfig
    ) {
  size_t idx = get_global_id(0);

  uint j = getJfrom2dIdx(idx, layersConfig);
  uint outputIdx = outputOffset + j;

  return 2 * (this->actvs[idx] - outputs[j]);
}

//===================================================================================================================//

kernel void calc_dCost_dActv(
    global TYPE* dCost_dAcvts,
    global TYPE* acvts,
    global TYPE* weights,
    global TYPE* zs,
    uint l,
    constant LayersConfig* layersConfig
    ) {
  size_t idx = get_global_id(0);

  uint k = getJfrom2dIdx(idx, layersConfig);

  const Layer& nextLayer = layersConfig[l + 1];
  uint nextNumNeurons = nextLayer.numNeurons;

  TYPE sum = 0;

  for (uint j = 0; j < nextNumNeurons; j++) {
    uint nextLayer2dIdx = get2dIdxFromLJ(l + 1, j);
    uint nextLayer3dIdx = get3dIdxFromLJK(l + 1, j, k, layersConfig);

    TYPE& weight = weights[nextLayer3dIdx];
    TYPE& z = zs[nextLayer2dIdx];

    ActvFuncType actvFuncType = layersConfig[l + 1].actvFuncType;
    TYPE dActvFunc_z = actvFunc_calculate(z, actvFuncType, true);

    TYPE& dCost_dActv = dCost_dAcvts[nextLayer2dIdx];

    sum += weight * dActvFunc_z * dConst_dActcv;
  }

  dCost_dActv[idx] = sum;
}

//===================================================================================================================//

kernel void calc_dCost_dWeight(
    global TYPE* dCost_dWeights,
    global TYPE* acvts,
    global TYPE* zs,
    global TYPE* dCost_dAcvts,
    uint l,
    constant LayersConfig* layersConfig
    ) {
  size_t idx = get_global_id(0);

  uint l, j k;
  getLJKfrom3dIdx(l, j, k, idx, layersConfig);

  uint prevLayer2dIdx = get2dIdxFromLJ(l - 1, k);
  TYPE& actv = this->actvs(prevLayer2dIdx);

  uint layer2dIdx = get2dIdxFromLJ(l, j);
  TYPE& z = zs[layer2dIdx];

  ActvFuncType actvFuncType = layersConfig[l].actvFuncType;
  TYPE dActvFunc_z = actvFunc_calculate(z, actvFuncType, true);

  TYPE& dCost_dActv = dCost_dAcvts[layer2dIdx];

  dCost_dWeights[idx] = actv * dActvFunc_z * dCost_dActv;
}

//===================================================================================================================//

kernel void calc_dCost_Bias(
    global TYPE* dCost_dBiases,
    global TYPE* zs,
    global TYPE* dCost_dActvs,
    uint l,
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
