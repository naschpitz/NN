#include "Common_LRSchedulerConfig.hpp"

#include <stdexcept>

namespace Common
{
  LRSchedulerType LRSchedulerConfig::nameToType(const std::string& name)
  {
    if (name == "none")
      return LRSchedulerType::NONE;

    if (name == "step")
      return LRSchedulerType::STEP;

    if (name == "cosine")
      return LRSchedulerType::COSINE;

    if (name == "plateau")
      return LRSchedulerType::PLATEAU;

    throw std::runtime_error("Unknown LR scheduler type: " + name);
  }

  //===================================================================================================================//

  std::string LRSchedulerConfig::typeToName(LRSchedulerType type)
  {
    switch (type) {
    case LRSchedulerType::NONE:
      return "none";
    case LRSchedulerType::STEP:
      return "step";
    case LRSchedulerType::COSINE:
      return "cosine";
    case LRSchedulerType::PLATEAU:
      return "plateau";
    default:
      return "none";
    }
  }
}

//===================================================================================================================//
