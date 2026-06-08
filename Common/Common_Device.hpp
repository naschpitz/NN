#ifndef COMMON_DEVICE_HPP
#define COMMON_DEVICE_HPP

#include <stdexcept>
#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace Common
{
  enum class DeviceType { CPU, GPU };

  const std::unordered_map<std::string, DeviceType> deviceTypeMap = {
    {"cpu", DeviceType::CPU},
    {"gpu", DeviceType::GPU},
  };

  class Device
  {
    public:
      static DeviceType nameToType(const std::string& name)
      {
        auto it = deviceTypeMap.find(name);

        if (it != deviceTypeMap.end()) {
          return it->second;
        }

        throw std::runtime_error("Unknown device type: " + name);
      }

      static std::string typeToName(const DeviceType& deviceType)
      {
        for (const auto& pair : deviceTypeMap) {
          if (pair.second == deviceType) {
            return pair.first;
          }
        }

        throw std::runtime_error("Unknown device type enum value");
      }
  };
}

//===================================================================================================================//

#endif // COMMON_DEVICE_HPP
