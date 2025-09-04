#include "ANN_CoreType.hpp"

using namespace ANN;

//===================================================================================================================//

CoreTypeType CoreType::nameToType(const std::string& name) {
  auto it = coreTypeMap.find(name);

  if (it == coreTypeMap.end()) {
    return CoreTypeType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string CoreType::typeToName(const CoreTypeType& coreTypeType) {
  for (const auto& pair : coreTypeMap) {
    if (pair.second == coreTypeType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//
