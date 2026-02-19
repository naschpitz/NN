#include "ANN_RunMode.hpp"

using namespace ANN;

//===================================================================================================================//

RunModeType RunMode::nameToType(const std::string& name) {
  auto it = runModeMap.find(name);

  if (it == runModeMap.end()) {
    return RunModeType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string RunMode::typeToName(const RunModeType& runModeType) {
  for (const auto& pair : runModeMap) {
    if (pair.second == runModeType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//

