#ifndef ANN_MODE_HPP
#define ANN_MODE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class ModeType {
    TRAIN,
    PREDICT,
    TEST,
    UNKNOWN
  };

  const std::unordered_map<std::string, ModeType> modeMap = {
    {"train", ModeType::TRAIN},
    {"predict", ModeType::PREDICT},
    {"test", ModeType::TEST},
  };

  class Mode
  {
    public:
      static ModeType nameToType(const std::string& name);
      static std::string typeToName(const ModeType& modeType);
  };
}

//===================================================================================================================//

#endif // ANN_MODE_HPP

