#ifndef CNN_PARAMETERS_HPP
#define CNN_PARAMETERS_HPP

#include "CNN_Conv2DParameters.hpp"
#include "CNN_NormParameters.hpp"
#include "CNN_ResidualParameters.hpp"

#include <_Parameters.hpp>

#include <vector>

//===================================================================================================================//

namespace CNN
{
  using namespace Common;
  // All CNN parameters (conv layers + batch norm layers + residual projections +  dense parameters)
  template <typename T>
  struct Parameters {
      std::vector<ConvParameters<T>> convParams;         // One per conv layer
      std::vector<NormParameters<T>> normParams;         // One per norm layer
      std::vector<ResidualParameters<T>> residualParams;  // One per residual block with channel mismatch
      ANN::Parameters<T> denseParams;                     // Dense layer parameters (delegated to )
  };
}

//===================================================================================================================//

#endif // CNN_PARAMETERS_HPP
