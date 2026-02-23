#ifndef ANN_COSTFUNCTIONCONFIG_HPP
#define ANN_COSTFUNCTIONCONFIG_HPP

#include <string>
#include <stdexcept>
#include <vector>

//===================================================================================================================//

namespace ANN {
  enum class CostFunctionType : int {
    SQUARED_DIFFERENCE = 0,
    WEIGHTED_SQUARED_DIFFERENCE = 1
  };

  //-- String conversion helpers --//
  struct CostFunction {
    static CostFunctionType nameToType(const std::string& name) {
      if (name == "squaredDifference") return CostFunctionType::SQUARED_DIFFERENCE;
      if (name == "weightedSquaredDifference") return CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
      throw std::runtime_error("Unknown cost function type: " + name);
    }

    static std::string typeToName(CostFunctionType type) {
      switch (type) {
        case CostFunctionType::SQUARED_DIFFERENCE: return "squaredDifference";
        case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE: return "weightedSquaredDifference";
        default: return "squaredDifference";
      }
    }
  };

  template <typename T>
  struct CostFunctionConfig {
    CostFunctionType type = CostFunctionType::SQUARED_DIFFERENCE;
    std::vector<T> weights;  // Per-output-neuron weights (only used for WEIGHTED_SQUARED_DIFFERENCE)
  };
}

//===================================================================================================================//

#endif // ANN_COSTFUNCTIONCONFIG_HPP

