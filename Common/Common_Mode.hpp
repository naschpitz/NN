#ifndef COMMON_MODE_HPP
#define COMMON_MODE_HPP

#include <string>

//===================================================================================================================//

namespace Common
{
  enum class ModeType { TRAIN, PREDICT, TEST, CALIBRATE };

  class Mode
  {
    public:
      static ModeType nameToType(const std::string& name);
      static std::string typeToName(const ModeType& modeType);
  };
}

//===================================================================================================================//

#endif // COMMON_MODE_HPP
