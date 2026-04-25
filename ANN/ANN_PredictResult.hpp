#ifndef ANN_PREDICTRESULT_HPP
#define ANN_PREDICTRESULT_HPP

#include "ANN_Types.hpp"

#include <vector>

//===================================================================================================================//

namespace ANN
{
  // Result of a single prediction.
  // - output: post-activation values of the last layer (e.g. softmax probabilities).
  // - logits: pre-activation values (z) of the last layer — input to the final activation.
  //
  // Logits enable calibration / out-of-distribution detection scores (max-logit, logit-norm,
  // free-energy −log Σ exp(zᵢ)) that softmax discards because softmax is shift-invariant:
  // softmax(z) == softmax(z + c).
  template <typename T>
  struct PredictResult {
      Output<T> output;
      Logits<T> logits;
  };

  template <typename T>
  using PredictResults = std::vector<PredictResult<T>>;
}

//===================================================================================================================//

#endif // ANN_PREDICTRESULT_HPP
