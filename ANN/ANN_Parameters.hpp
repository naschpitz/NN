#ifndef ANN_PARAMETERS_HPP
#define ANN_PARAMETERS_HPP

#include "ANN_Types.hpp"

//===================================================================================================================//

namespace ANN {
  template <typename T>
  struct Parameters {
    Tensor3D<T> weights;
    Tensor2D<T> biases;
  };
}

//===================================================================================================================//

#endif // ANN_PARAMETERS_HPP

