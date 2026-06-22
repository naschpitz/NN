#ifndef COMMON_LRSCHEDULER_HPP
#define COMMON_LRSCHEDULER_HPP

#include "Common_LRSchedulerConfig.hpp"

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  //-- Mutable runtime state. Persisted in the checkpoint so resumed runs continue the schedule. --//
  struct LRSchedulerState {
      float baseLR = 0.0f; // Original base LR (captured at run start; the step/cosine reference)
      float currentLR = 0.0f; // Last LR applied
      ulong epochsSinceImprovement = 0; // Plateau counter
      float bestValLoss = 0.0f; // Plateau best (use +inf sentinel on fresh start)
      bool initialized = false; // Has the scheduler seen its first epoch
  };

  //===================================================================================================================//
  //--
  // Pure LR-scheduler step. Returns the new currentLR and updates `state` in place.
  // `epoch` is absolute (startingEpoch + epochsDone) so cosine continues across resumes.
  // For NONE, state and currentLR are untouched (flat-LR backward-compatible behavior).
  //--
  float stepLRScheduler(const LRSchedulerConfig& cfg, LRSchedulerState& state, ulong epoch, ulong totalEpochs,
                        bool hasValLoss, float valLoss);
}

//===================================================================================================================//

#endif // COMMON_LRSCHEDULER_HPP
