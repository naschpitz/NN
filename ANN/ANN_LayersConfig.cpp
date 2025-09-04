#include "ANN_LayersConfig.hpp"

using namespace ANN;

//===================================================================================================================//

ulong LayersConfig::getTotalNumNeurons() const {
  ulong sum = 0;

  for (Layer layer : *this) {
    sum += layer.numNeurons;
  }

  return sum;
}

//===================================================================================================================//
