#include "ANN_Worker.hpp"

#include <algorithm>
#include <cmath>

using namespace ANN;

//===================================================================================================================//

template <typename T>
T Worker<T>::calculateLoss(const Output<T>& predicted, const Output<T>& expected) {
  T loss = 0;

  switch (this->costFunctionConfig.type) {
    case CostFunctionType::CROSS_ENTROPY: {
      // Cross-entropy: L = -sum(w_i * y_i * log(a_i))
      const T epsilon = static_cast<T>(1e-7);
      for (ulong i = 0; i < expected.size(); i++) {
        T pred = std::max(predicted[i], epsilon);
        T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
        loss -= weight * expected[i] * std::log(pred);
      }
      break;
    }

    case CostFunctionType::SQUARED_DIFFERENCE:
    case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
    default: {
      // Squared difference: L = sum(w_i * (a_i - y_i)^2) / N
      for (ulong i = 0; i < expected.size(); i++) {
        T diff = predicted[i] - expected[i];
        T weight = (!this->costFunctionConfig.weights.empty()) ? this->costFunctionConfig.weights[i] : static_cast<T>(1);
        loss += weight * diff * diff;
      }
      loss /= static_cast<T>(expected.size());
      break;
    }
  }

  return loss;
}

//===================================================================================================================//

// Explicit template instantiations.
template class ANN::Worker<int>;
template class ANN::Worker<double>;
template class ANN::Worker<float>;

