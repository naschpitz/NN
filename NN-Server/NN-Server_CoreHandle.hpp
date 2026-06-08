#pragma once

#include <_Core.hpp>
#include <CNN_Core.hpp>

namespace NN_Server
{

  struct CoreHandle {
      ANN::Core<float>* annCore = nullptr;
      CNN::Core<float>* cnnCore = nullptr;
      int index = -1;
  };

} // namespace NN_Server
