#include "Common_LRScheduler.hpp"

#include <algorithm>
#include <cmath>

namespace Common
{
  float stepLRScheduler(const LRSchedulerConfig& cfg, LRSchedulerState& state, ulong epoch, ulong totalEpochs,
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
