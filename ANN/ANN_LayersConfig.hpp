#ifndef LAYERSCONFIG_HPP
#define LAYERSCONFIG_HPP

#include "_ActvFunc.hpp"

#include <vector>

namespace ANN
{
  using namespace Common;
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
