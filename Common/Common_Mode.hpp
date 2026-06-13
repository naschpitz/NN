#ifndef COMMON_MODE_HPP
#define COMMON_MODE_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace Common
{
  enum class ModeType { TRAIN, PREDICT, TEST, CALIBRATE };

  const std::unordered_map<std::string, ModeType> modeMap = {
    {"train", ModeType::TRAIN},
    {"predict", ModeType::PREDICT},
    {"test", ModeType::TEST},
    {"calibrate", ModeType::CALIBRATE},
  };

  class Mode
  {
    public:
      static ModeType nameToType(const std::string& name)
      {
        auto it = modeMap.find(name);

        if (it != modeMap.end()) {
          return it->second;
        }

        throw std::runtime_error("Unknown mode type: " + name);
      }

      static std::string typeToName(const ModeType& modeType)
      {
        for (const auto& pair : modeMap) {
          if (pair.second == modeType) {
            return pair.first;
          }
        }

        throw std::runtime_error("Unknown mode type enum value");
      }
  };
}

//===================================================================================================================//

#endif // COMMON_MODE_HPP
