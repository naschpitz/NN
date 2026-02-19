#include "ANN_Mode.hpp"

using namespace ANN;

//===================================================================================================================//

ModeType Mode::nameToType(const std::string& name) {
  auto it = modeMap.find(name);

  if (it == modeMap.end()) {
    return ModeType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string Mode::typeToName(const ModeType& modeType) {
  for (const auto& pair : modeMap) {
    if (pair.second == modeType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//

