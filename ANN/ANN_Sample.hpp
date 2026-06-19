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

  // Non-owning view over a sequence of samples (e.g. a sub-batch handed to a worker).
  template <typename T>
  using SamplesView = std::span<const Sample<T>>;
}

//===================================================================================================================//

#endif // ANN_SAMPLE_HPP
