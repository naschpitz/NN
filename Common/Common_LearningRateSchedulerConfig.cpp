#include "Common_LearningRateSchedulerConfig.hpp"

#include <stdexcept>

namespace Common
{
  LearningRateSchedulerType LearningRateSchedulerConfig::nameToType(const std::string& name)
  {
    if (name == "none")
      return LearningRateSchedulerType::NONE;

    if (name == "step")
      return LearningRateSchedulerType::STEP;

    if (name == "cosine")
      return LearningRateSchedulerType::COSINE;

    if (name == "plateau")
      return LearningRateSchedulerType::PLATEAU;

    throw std::runtime_error("Unknown LR scheduler type: " + name);
  }

  //===================================================================================================================//

  std::string LearningRateSchedulerConfig::typeToName(LearningRateSchedulerType type)
  {
    switch (type) {
    case LearningRateSchedulerType::NONE:
      return "none";
    case LearningRateSchedulerType::STEP:
      return "step";
    case LearningRateSchedulerType::COSINE:
      return "cosine";
    case LearningRateSchedulerType::PLATEAU:
      return "plateau";
    default:
      return "none";
    }
  }
}

//===================================================================================================================//
