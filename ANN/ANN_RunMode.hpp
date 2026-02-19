#ifndef ANN_RUNMODE_HPP
#define ANN_RUNMODE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class RunModeType {
    TRAIN,
    RUN,
    TEST,
    UNKNOWN
  };

  const std::unordered_map<std::string, RunModeType> runModeMap = {
    {"train", RunModeType::TRAIN},
    {"run", RunModeType::RUN},
    {"test", RunModeType::TEST},
  };

  class RunMode
  {
    public:
      static RunModeType nameToType(const std::string& name);
      static std::string typeToName(const RunModeType& runModeType);
  };
}

//===================================================================================================================//

#endif // ANN_RUNMODE_HPP

