#ifndef CORETYPE_HPP
#define CORETYPE_HPP

#include <string>
#include <unordered_map>

//===================================================================================================================//

namespace ANN {
  enum class CoreTypeType {
    CPU,
    GPU,
    UNKNOWN
  };

  const std::unordered_map<std::string, CoreTypeType> coreTypeMap = {
    {"cpu", CoreTypeType::CPU},
    {"gpu", CoreTypeType::GPU},
  };

  class CoreType
  {
    public:
      static CoreTypeType nameToType(const std::string& name);
      static std::string typeToName(const CoreTypeType& actvFuncType);
  };
}

#endif // CORETYPE_HPP
