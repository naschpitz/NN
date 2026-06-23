#ifndef COMMON_OPTIMIZER_HPP
#define COMMON_OPTIMIZER_HPP

#include <string>

//===================================================================================================================//

namespace Common
{
  enum class OptimizerType : int { SGD = 0, ADAM = 1 };

  //-- Name/type conversion. Type-independent, so these are free functions rather than
  // template members of Optimizer<T> (which would force the definitions back into the header). --//
  OptimizerType optimizerNameToType(const std::string& name);
  std::string optimizerTypeToName(OptimizerType t);

  template <typename T>
  struct Optimizer {
      //-- Configuration --//
      OptimizerType type = OptimizerType::SGD;
      T beta1 = static_cast<T>(0.9);
      T beta2 = static_cast<T>(0.999);
      T epsilon = static_cast<T>(1e-8);
  };
}

//===================================================================================================================//

#endif // COMMON_OPTIMIZER_HPP
