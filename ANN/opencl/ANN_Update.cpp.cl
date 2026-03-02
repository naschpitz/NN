#ifndef ANN_UPDATE_CPP_CL
#define ANN_UPDATE_CPP_CL

// Note: Depends on ANN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

kernel void accumulate_dCost_dBiases(global TYPE* accum_dCost_dBiases, global TYPE* dCost_dBiases, ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    accum_dCost_dBiases[idx] += dCost_dBiases[idx];
  }
}

//===================================================================================================================//

kernel void accumulate_dCost_dWeights(global TYPE* accum_dCost_dWeights, global TYPE* dCost_dWeights, ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    accum_dCost_dWeights[idx] += dCost_dWeights[idx];
  }
}

//===================================================================================================================//

kernel void update_biases(global TYPE* biases, global TYPE* accum_dCost_dBiases, ulong numSamples, float learningRate,
                          ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    biases[idx] -= learningRate * (accum_dCost_dBiases[idx] / (TYPE)numSamples);
  }
}

//===================================================================================================================//

kernel void update_weights(global TYPE* weights, global TYPE* accum_dCost_dWeights, ulong numSamples,
                           float learningRate, ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    weights[idx] -= learningRate * (accum_dCost_dWeights[idx] / (TYPE)numSamples);
  }
}

//===================================================================================================================//

kernel void update_biases_adam(global TYPE* biases, global TYPE* accum_dCost_dBiases, global TYPE* m, global TYPE* v,
                               ulong numSamples, float learningRate, float beta1, float beta2, float epsilon, float bc1,
                               float bc2, ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    TYPE g = accum_dCost_dBiases[idx] / (TYPE)numSamples;
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    TYPE m_hat = m[idx] / bc1;
    TYPE v_hat = v[idx] / bc2;
    biases[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
  }
}

//===================================================================================================================//

kernel void update_weights_adam(global TYPE* weights, global TYPE* accum_dCost_dWeights, global TYPE* m, global TYPE* v,
                                ulong numSamples, float learningRate, float beta1, float beta2, float epsilon,
                                float bc1, float bc2, ulong size)
{
  size_t idx = get_global_id(0);

  if (idx < size) {
    TYPE g = accum_dCost_dWeights[idx] / (TYPE)numSamples;
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    TYPE m_hat = m[idx] / bc1;
    TYPE v_hat = v[idx] / bc2;
    weights[idx] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
  }
}

//===================================================================================================================//

#endif // ANN_UPDATE_CPP_CL
