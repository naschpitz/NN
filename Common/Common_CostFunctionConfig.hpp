#ifndef COMMON_COSTFUNCTIONCONFIG_HPP
#define COMMON_COSTFUNCTIONCONFIG_HPP

#include <string>
#include <vector>

//===================================================================================================================//

namespace Common
{
  enum class CostFunctionType : int { SQUARED_DIFFERENCE = 0, WEIGHTED_SQUARED_DIFFERENCE = 1, CROSS_ENTROPY = 2 };

  //-- String conversion helpers --//
  struct CostFunction {
      static CostFunctionType nameToType(const std::string& name);
      static std::string typeToName(CostFunctionType type);
  };

  template <typename T>
  struct CostFunctionConfig {
      CostFunctionType type = CostFunctionType::SQUARED_DIFFERENCE;
      std::vector<T> weights; // Per-output-neuron weights (only used for WEIGHTED_SQUARED_DIFFERENCE)
  };
}

//===================================================================================================================//

#endif // COMMON_COSTFUNCTIONCONFIG_HPP
