#ifndef COMMON_DEVICE_HPP
#define COMMON_DEVICE_HPP

#include <string>

//===================================================================================================================//

namespace Common
{
  enum class DeviceType { CPU, GPU };

  class Device
  {
    public:
      static DeviceType nameToType(const std::string& name);
      static std::string typeToName(const DeviceType& deviceType);
  };
}

//===================================================================================================================//

#endif // COMMON_DEVICE_HPP
