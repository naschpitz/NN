#include "Common_Mode.hpp"

#include <stdexcept>
#include <unordered_map>

namespace Common
{
  namespace
  {
    const std::unordered_map<std::string, ModeType> modeMap = {
      {"train", ModeType::TRAIN},
      {"predict", ModeType::PREDICT},
      {"test", ModeType::TEST},
      {"calibrate", ModeType::CALIBRATE},
    };
  }

  ModeType Mode::nameToType(const std::string& name)
  {
    auto it = modeMap.find(name);

    if (it != modeMap.end()) {
      return it->second;
    }

    throw std::runtime_error("Unknown mode type: " + name);
  }

  //===================================================================================================================//

  std::string Mode::typeToName(const ModeType& modeType)
  {
    for (const auto& pair : modeMap) {
      if (pair.second == modeType) {
        return pair.first;
      }
    }

    throw std::runtime_error("Unknown mode type enum value");
  }
}

//===================================================================================================================//
