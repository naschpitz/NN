#ifndef _PARAMETERS_HPP
#define _PARAMETERS_HPP

#include "_Types.hpp"

//===================================================================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  struct Parameters {
      Tensor3D<T> weights;
      Tensor2D<T> biases;
  };
}

//===================================================================================================================//

#endif // _PARAMETERS_HPP
