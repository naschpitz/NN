#ifndef COMMON_OPTIMIZER_HPP
#define COMMON_OPTIMIZER_HPP

#include <stdexcept>
#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  enum class OptimizerType : int { SGD = 0, ADAM = 1 };

  template <typename T>
  struct Optimizer {
      //-- Configuration --//
      OptimizerType type = OptimizerType::SGD;
      T beta1 = static_cast<T>(0.9);
      T beta2 = static_cast<T>(0.999);
      T epsilon = static_cast<T>(1e-8);

      //-- Name/type conversion --//
      static OptimizerType nameToType(const std::string& name)
      {
        if (name == "sgd")
          return OptimizerType::SGD;

        if (name == "adam")
          return OptimizerType::ADAM;

        throw std::runtime_error("Unknown optimizer type: " + name);
      }

      static std::string typeToName(OptimizerType t)
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
  };
}

//===================================================================================================================//

#endif // COMMON_OPTIMIZER_HPP
