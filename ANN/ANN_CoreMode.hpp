#ifndef ANN_COREMODE_H
#define ANN_COREMODE_H

#include <string>
#include <unordered_map>

namespace ANN {
  enum class CoreModeType {
    TRAIN,
    RUN,
    UNKNOWN
  };

  const std::unordered_map<std::string, CoreModeType> coreMap = {
    {"train", CoreModeType::TRAIN},
    {"run", CoreModeType::RUN},
  };

  class CoreMode
  {
    public:
      static CoreModeType nameToType(const std::string& name);
      static std::string typeToName(const CoreModeType& actvFuncType);
  };
}

#endif // ANN_COREMODE_H
