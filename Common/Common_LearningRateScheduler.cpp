#include "Common_LearningRateScheduler.hpp"

#include <algorithm>
#include <cmath>

namespace Common
{
  float stepLearningRateScheduler(const LearningRateSchedulerConfig& cfg, LearningRateSchedulerState& state,
                                  ulong epoch, ulong totalEpochs, bool hasValLoss, float valLoss)
  {
    constexpr float kPi = 3.14159265358979323846f;

    switch (cfg.type) {
    case LearningRateSchedulerType::NONE:
      return state.currentLearningRate;

    case LearningRateSchedulerType::STEP: {
      const ulong steps = (cfg.stepSize == 0) ? 0 : epoch / cfg.stepSize;
      const float factor = std::pow(cfg.gamma, static_cast<float>(steps));
      const float learningRate = state.baseLearningRate * factor;
      return std::max(learningRate, cfg.minLearningRate);
    }

    case LearningRateSchedulerType::COSINE: {
      const float denom = (totalEpochs == 0) ? 1.0f : static_cast<float>(totalEpochs);
      float progress = static_cast<float>(epoch) / denom;
      progress = std::min(progress, 1.0f);
      const float learningRate =
        cfg.minLearningRate + 0.5f * (state.baseLearningRate - cfg.minLearningRate) * (1.0f + std::cos(kPi * progress));
      return std::max(learningRate, cfg.minLearningRate);
    }

    case LearningRateSchedulerType::PLATEAU: {
      if (!state.initialized) {
        state.bestValidationLoss = valLoss;
        state.epochsSinceImprovement = 0;
        state.initialized = true;
        return state.currentLearningRate;
      }

      if (hasValLoss && (valLoss < state.bestValidationLoss - cfg.minDelta)) {
        state.bestValidationLoss = valLoss;
        state.epochsSinceImprovement = 0;
      } else {
        ++state.epochsSinceImprovement;
      }

      if (state.epochsSinceImprovement >= cfg.patience) {
        const float learningRate = std::max(state.currentLearningRate * cfg.gamma, cfg.minLearningRate);
        state.epochsSinceImprovement = 0;
        return learningRate;
      }

      return state.currentLearningRate;
    }

    default:
      return state.currentLearningRate;
    }
  }
}

//===================================================================================================================//
