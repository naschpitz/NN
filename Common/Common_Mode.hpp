#ifndef COMMON_MODE_HPP
#define COMMON_MODE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace Common
{
  enum class ModeType { TRAIN, PREDICT, TEST };

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

#endif // COMMON_MODE_HPP
