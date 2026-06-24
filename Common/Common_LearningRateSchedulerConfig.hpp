#ifndef COMMON_LEARNINGRATESCHEDULER_CONFIG_HPP
#define COMMON_LEARNINGRATESCHEDULER_CONFIG_HPP

#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  enum class LearningRateSchedulerType : int { NONE = 0, STEP = 1, COSINE = 2, PLATEAU = 3 };

  class LearningRateSchedulerConfig
  {
    public:
      //-- Configuration fields --//
      LearningRateSchedulerType type = LearningRateSchedulerType::NONE;
      float gamma = 0.1f; // Multiplicative factor (step, plateau)
      ulong stepSize = 1; // Epochs per decay step (step)
      float minLearningRate = 0.0f; // Lower bound (cosine, plateau)
      ulong patience = 10; // Epochs w/o improvement before reduce (plateau)
      float minDelta = 1e-4f; // Improvement threshold (plateau)

      //-- Name/type conversion --//
      static LearningRateSchedulerType nameToType(const std::string& name);
      static std::string typeToName(LearningRateSchedulerType type);
  };
}

//===================================================================================================================//

#endif // COMMON_LEARNINGRATESCHEDULER_CONFIG_HPP
