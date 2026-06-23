#include "Common_Optimizer.hpp"

#include <stdexcept>

namespace Common
{
  OptimizerType optimizerNameToType(const std::string& name)
  {
    if (name == "sgd")
      return OptimizerType::SGD;

    if (name == "adam")
      return OptimizerType::ADAM;

    throw std::runtime_error("Unknown optimizer type: " + name);
  }

  //===================================================================================================================//

  std::string optimizerTypeToName(OptimizerType t)
  {
    switch (t) {
    case OptimizerType::SGD:
      return "sgd";
    case OptimizerType::ADAM:
      return "adam";
    default:
      return "sgd";
    }
  }
}

//===================================================================================================================//
