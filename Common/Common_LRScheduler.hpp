#ifndef COMMON_LRSCHEDULER_HPP
#define COMMON_LRSCHEDULER_HPP

#include <stdexcept>
#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  enum class LRSchedulerType : int { NONE = 0, STEP = 1, COSINE = 2, PLATEAU = 3 };

  struct LRSchedulerConfig {
      LRSchedulerType type = LRSchedulerType::NONE;
      float gamma = 0.1f; // Multiplicative factor (step, plateau)
      ulong stepSize = 1; // Epochs per decay step (step)
      float minLR = 0.0f; // Lower bound (cosine, plateau)
      ulong patience = 10; // Epochs w/o improvement before reduce (plateau)
      float minDelta = 1e-4f; // Improvement threshold (plateau)

      //-- Name/type conversion --//
      static LRSchedulerType nameToType(const std::string& name)
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

      static std::string typeToName(LRSchedulerType type)
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
  };

  //-- Mutable runtime state. Persisted in the checkpoint so resumed runs continue the schedule. --//
  struct LRSchedulerState {
      float baseLR = 0.0f; // Original base LR (captured at run start; the step/cosine reference)
      float currentLR = 0.0f; // Last LR applied
      ulong epochsSinceImprovement = 0; // Plateau counter
      float bestValLoss = 0.0f; // Plateau best (use +inf sentinel on fresh start)
      bool initialized = false; // Has the scheduler seen its first epoch
  };
}

//===================================================================================================================//

#endif // COMMON_LRSCHEDULER_HPP
