#include "_LayersConfig.hpp"

using namespace ANN;
using namespace Common;

//===================================================================================================================//

ulong LayersConfig::getTotalNumNeurons() const
{
  ulong sum = 0;

  for (Layer layer : *this) {
    sum += layer.numNeurons;
  }

  return sum;
}

//===================================================================================================================//
