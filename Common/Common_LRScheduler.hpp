#ifndef COMMON_LRSCHEDULER_HPP
#define COMMON_LRSCHEDULER_HPP

#include <algorithm>
#include <cmath>
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

  //===================================================================================================================//
  //--
  // Pure LR-scheduler step. Returns the new currentLR and updates `state` in place.
  // `epoch` is absolute (startingEpoch + epochsDone) so cosine continues across resumes.
  // For NONE, state and currentLR are untouched (flat-LR backward-compatible behavior).
  //--
  inline float stepLRScheduler(const LRSchedulerConfig& cfg, LRSchedulerState& state, ulong epoch, ulong totalEpochs,
                               bool hasValLoss, float valLoss)
  {
    constexpr float kPi = 3.14159265358979323846f;

    switch (cfg.type) {
    case LRSchedulerType::NONE:
      return state.currentLR;

    case LRSchedulerType::STEP: {
      const ulong steps = (cfg.stepSize == 0) ? 0 : epoch / cfg.stepSize;
      const float factor = std::pow(cfg.gamma, static_cast<float>(steps));
      const float lr = state.baseLR * factor;
      return std::max(lr, cfg.minLR);
    }

    case LRSchedulerType::COSINE: {
      const float denom = (totalEpochs == 0) ? 1.0f : static_cast<float>(totalEpochs);
      float progress = static_cast<float>(epoch) / denom;
      progress = std::min(progress, 1.0f);
      const float lr = cfg.minLR + 0.5f * (state.baseLR - cfg.minLR) * (1.0f + std::cos(kPi * progress));
      return std::max(lr, cfg.minLR);
    }

    case LRSchedulerType::PLATEAU: {
      if (!state.initialized) {
        state.bestValLoss = valLoss;
        state.epochsSinceImprovement = 0;
        state.initialized = true;
        return state.currentLR;
      }

      if (hasValLoss && (valLoss < state.bestValLoss - cfg.minDelta)) {
        state.bestValLoss = valLoss;
        state.epochsSinceImprovement = 0;
      } else {
        ++state.epochsSinceImprovement;
      }

      if (state.epochsSinceImprovement >= cfg.patience) {
        const float lr = std::max(state.currentLR * cfg.gamma, cfg.minLR);
        state.epochsSinceImprovement = 0;
        return lr;
      }

      return state.currentLR;
    }

    default:
      return state.currentLR;
    }
  }
}

//===================================================================================================================//

#endif // COMMON_LRSCHEDULER_HPP
