#include "ANN_Device.hpp"

using namespace ANN;

//===================================================================================================================//

DeviceType Device::nameToType(const std::string& name) {
  auto it = deviceTypeMap.find(name);

  if (it == deviceTypeMap.end()) {
    return DeviceType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string Device::typeToName(const DeviceType& deviceType) {
  for (const auto& pair : deviceTypeMap) {
    if (pair.second == deviceType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//

