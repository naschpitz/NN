#ifndef COMMON_LEARNINGRATESCHEDULER_HPP
#define COMMON_LEARNINGRATESCHEDULER_HPP

#include "Common_LearningRateSchedulerConfig.hpp"

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  //-- Mutable runtime state. Persisted in the checkpoint so resumed runs continue the schedule. --//
  struct LearningRateSchedulerState {
      float baseLearningRate = 0.0f; // Original base learning rate (captured at run start; the step/cosine reference)
      float currentLearningRate = 0.0f; // Last learning rate applied
      ulong epochsSinceImprovement = 0; // Plateau counter
      float bestValidationLoss = 0.0f; // Plateau best (use +inf sentinel on fresh start)
      bool initialized = false; // Has the scheduler seen its first epoch
  };

  //===================================================================================================================//
  //--
  // Pure learning-rate scheduler step. Returns the new currentLearningRate and updates `state` in place.
  // `epoch` is absolute (startingEpoch + epochsDone) so cosine continues across resumes.
  // For NONE, state and currentLearningRate are untouched (flat-learning-rate backward-compatible behavior).
  //--
  float stepLearningRateScheduler(const LearningRateSchedulerConfig& cfg, LearningRateSchedulerState& state,
                                  ulong epoch, ulong totalEpochs, bool hasValLoss, float valLoss);
}

//===================================================================================================================//

#endif // COMMON_LEARNINGRATESCHEDULER_HPP
