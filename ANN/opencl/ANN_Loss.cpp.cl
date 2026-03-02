#ifndef ANN_LOSS_CPP_CL
#define ANN_LOSS_CPP_CL

// Note: Depends on ANN_Defines.hpp.cl (TYPE)

//===================================================================================================================//

// calculate_sample_loss: computes weighted loss on GPU and accumulates it.
// Single work-item kernel — numOutputs is small (e.g. 11).
// costFunctionType: 0/1 = MSE, 2 = cross-entropy
kernel void calculate_sample_loss(global TYPE* actvs, global TYPE* outputs, global TYPE* lossWeights,
                                  global TYPE* accumLoss, ulong actvOffset, ulong numOutputs, ulong costFunctionType)
{
  TYPE loss = (TYPE)0;

  if (costFunctionType == 2) {
    // Cross-entropy: L = -sum(w_i * y_i * log(max(a_i, epsilon)))
    for (ulong i = 0; i < numOutputs; i++) {
      TYPE predicted = actvs[actvOffset + i];

      if (predicted < (TYPE)1e-7)
        predicted = (TYPE)1e-7;
      loss -= lossWeights[i] * outputs[i] * log(predicted);
    }
  } else {
    // MSE: L = sum(w_i * (a_i - y_i)^2) / numOutputs
    for (ulong i = 0; i < numOutputs; i++) {
      TYPE diff = actvs[actvOffset + i] - outputs[i];
      loss += lossWeights[i] * diff * diff;
    }

    loss /= (TYPE)numOutputs;
  }

  accumLoss[0] += loss;
}

//===================================================================================================================//

#endif // ANN_LOSS_CPP_CL
