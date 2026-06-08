#ifndef ANN_SAMPLE_HPP
#define ANN_SAMPLE_HPP

#include "ANN_Types.hpp"
#include "Common/Common_Device.hpp"

//===================================================================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  struct Sample {
      Input<T> input;
      Output<T> output;
  };

  template <typename T>
  using Samples = std::vector<Sample<T>>;
}

//===================================================================================================================//

#endif // ANN_SAMPLE_HPP
