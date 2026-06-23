#include "Common_CostFunctionConfig.hpp"

#include <stdexcept>

namespace Common
{
  CostFunctionType CostFunction::nameToType(const std::string& name)
  {
    if (name == "squaredDifference")
      return CostFunctionType::SQUARED_DIFFERENCE;

    if (name == "weightedSquaredDifference")
      return CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;

    if (name == "crossEntropy")
      return CostFunctionType::CROSS_ENTROPY;
    throw std::runtime_error("Unknown cost function type: " + name);
  }

  //===================================================================================================================//

  std::string CostFunction::typeToName(CostFunctionType type)
  {
    switch (type) {
    case CostFunctionType::SQUARED_DIFFERENCE:
      return "squaredDifference";
    case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
      return "weightedSquaredDifference";
    case CostFunctionType::CROSS_ENTROPY:
      return "crossEntropy";
    default:
      return "squaredDifference";
    }
  }
}

//===================================================================================================================//
