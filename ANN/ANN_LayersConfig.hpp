#ifndef LAYERSCONFIG_HPP
#define LAYERSCONFIG_HPP

#include "ANN_ActvFunc.hpp"

#include <vector>

namespace ANN {
  struct Layer {
    ulong numNeurons;
    ActvFuncType actvFuncType;
  };

  class LayersConfig : public std::vector<Layer>
  {
    public:
      ulong getTotalNumNeurons() const;
  };
}

#endif // LAYERSCONFIG_HPP
