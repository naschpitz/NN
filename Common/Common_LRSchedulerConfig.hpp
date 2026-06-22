#ifndef COMMON_LRSCHEDULER_CONFIG_HPP
#define COMMON_LRSCHEDULER_CONFIG_HPP

#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  enum class LRSchedulerType : int { NONE = 0, STEP = 1, COSINE = 2, PLATEAU = 3 };

  class LRSchedulerConfig
  {
    public:
      //-- Configuration fields --//
      LRSchedulerType type = LRSchedulerType::NONE;
      float gamma = 0.1f; // Multiplicative factor (step, plateau)
      ulong stepSize = 1; // Epochs per decay step (step)
      float minLR = 0.0f; // Lower bound (cosine, plateau)
      ulong patience = 10; // Epochs w/o improvement before reduce (plateau)
      float minDelta = 1e-4f; // Improvement threshold (plateau)

      //-- Name/type conversion --//
      static LRSchedulerType nameToType(const std::string& name);
      static std::string typeToName(LRSchedulerType type);
  };
}

//===================================================================================================================//

#endif // COMMON_LRSCHEDULER_CONFIG_HPP
