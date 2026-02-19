#ifndef ANN_COREMODE_H
#define ANN_COREMODE_H

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class CoreModeType {
    TRAIN,
    RUN,
    TEST,
    UNKNOWN
  };

  const std::unordered_map<std::string, CoreModeType> coreModeMap = {
    {"train", CoreModeType::TRAIN},
    {"run", CoreModeType::RUN},
    {"test", CoreModeType::TEST},
  };

  class CoreMode
  {
    public:
      static CoreModeType nameToType(const std::string& name);
      static std::string typeToName(const CoreModeType& actvFuncType);
  };
}

//===================================================================================================================//

#endif // ANN_COREMODE_H
