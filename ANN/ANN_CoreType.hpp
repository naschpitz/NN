#ifndef CORETYPE_HPP
#define CORETYPE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class DeviceType {
    CPU,
    GPU,
    UNKNOWN
  };

  const std::unordered_map<std::string, DeviceType> deviceTypeMap = {
    {"cpu", DeviceType::CPU},
    {"gpu", DeviceType::GPU},
  };

  class CoreType
  {
    public:
      static DeviceType nameToType(const std::string& name);
      static std::string typeToName(const DeviceType& deviceType);
  };
}

#endif // CORETYPE_HPP
