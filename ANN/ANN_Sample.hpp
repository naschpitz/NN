#ifndef _SAMPLE_HPP
#define _SAMPLE_HPP

#include "_Types.hpp"

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

#endif // _SAMPLE_HPP
