#ifndef ANN_DEVICE_HPP
#define ANN_DEVICE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class DeviceType {
    CPU,
    GPU
  };

  const std::unordered_map<std::string, DeviceType> deviceTypeMap = {
    {"cpu", DeviceType::CPU},
    {"gpu", DeviceType::GPU},
  };

  class Device
  {
    public:
      static DeviceType nameToType(const std::string& name);
      static std::string typeToName(const DeviceType& deviceType);
  };
}

//===================================================================================================================//

#endif // ANN_DEVICE_HPP

