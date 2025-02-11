#ifndef ANN_ACTVFUNC_H
#define ANN_ACTVFUNC_H

#include <string>
#include <unordered_map>

namespace ANN {
  enum class ActvFuncType {
    RELU,
    SIGMOID,
    TANH,
    UNKNOWN
  };

  const std::unordered_map<std::string, ActvFuncType> actvMap = {
    {"relu", ActvFuncType::RELU},
    {"sigmoid", ActvFuncType::SIGMOID},
    {"tanh", ActvFuncType::TANH}
  };

  class ActvFunc {
    public:
      static ActvFuncType nameToType(const std::string& name);
      static std::string typeToName(const ActvFuncType& actvFuncType);

      static float calculate(float x, ActvFuncType type, bool derivative = false);

    private:
      static float relu(float x);
      static float sigmoid(float x);
      static float tanh(float x);

      static float drelu(float x);
      static float dsigmoid(float x);
      static float dtanh(float x);
  };
}

#endif // ANN_ACTVFUNC_H
