#include "ANN_CoreMode.hpp"

using namespace ANN;

//===================================================================================================================//

CoreModeType CoreMode::nameToType(const std::string& name) {
  auto it = coreMap.find(name);

  if (it == coreMap.end()) {
    return CoreModeType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string CoreMode::typeToName(const CoreModeType& coreModeType) {
  for (const auto& pair : coreMap) {
    if (pair.second == coreModeType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//
