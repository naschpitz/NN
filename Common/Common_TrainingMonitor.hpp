#ifndef COMMON_TRAININGMONITOR_HPP
#define COMMON_TRAININGMONITOR_HPP

#include "Common_MonitoringConfig.hpp"

#include <cmath>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  template <typename T>
  class TrainingMonitor
  {
    public:
      TrainingMonitor(const MonitoringConfig& config)
        : config(config),
          bestLoss(std::numeric_limits<T>::max()),
          bestEpoch(0),
          epochsWithoutImprovement(0),
          newBest(false),
          stopReason()
      {
      }

      // Called at end of each epoch. Returns true if training should stop.
      bool checkEpoch(ulong epoch, T epochLoss, std::optional<T> validationLoss = std::nullopt)
      {
        this->newBest = false;

        // Use validation loss when available, otherwise training loss
        T loss = validationLoss.has_value() ? validationLoss.value() : epochLoss;

        // Update best loss tracking
        T delta = static_cast<T>(config.metrics.lossStagnation.minDelta);

        if (loss < this->bestLoss - delta) {
          this->bestLoss = loss;
          this->bestEpoch = epoch;
          this->newBest = true;
          epochsWithoutImprovement = 0;
        }

        // Only perform metric checks at checkInterval boundaries
        if (epoch % config.checkInterval != 0) {
          return false;
        }

        // Check loss explosion (immediate stop)
        if (checkLossExplosion(loss)) {
          return true;
        }

        // Check loss stagnation (patience-based)
        if (!this->newBest) {
          if (checkLossStagnation(loss)) {
            return true;
          }
        }

        return false;
      }

      //-- Accessors --//
      bool isNewBest() const
      {
        return this->newBest;
      }
      std::string getStopReason() const
      {
        return this->stopReason;
      }
      ulong getBestEpoch() const
      {
        return this->bestEpoch;
      }
      T getBestLoss() const
      {
        return this->bestLoss;
      }

    private:
      //-- Configuration --//
      MonitoringConfig config;

      //-- Loss tracking --//
      T bestLoss;
      ulong bestEpoch;
      ulong epochsWithoutImprovement;
      bool newBest;
      std::string stopReason;

      //-- Metric evaluators --//
      bool checkLossStagnation(T loss)
      {
        if (!config.metrics.lossStagnation.enabled) {
          return false;
        }

        epochsWithoutImprovement++;

        if (epochsWithoutImprovement >= config.patience) {
          std::ostringstream oss;
          oss << "Loss stagnation: no improvement > " << config.metrics.lossStagnation.minDelta << " for "
              << config.patience << " check intervals (" << config.patience * config.checkInterval << " epochs)";
          this->stopReason = oss.str();
          return true;
        }

        return false;
      }

      bool checkLossExplosion(T loss)
      {
        if (!config.metrics.lossExplosion.enabled) {
          return false;
        }

        if (std::isnan(static_cast<double>(loss))) {
          this->stopReason = "Loss explosion: NaN detected";
          return true;
        }

        // Only check threshold if we have a valid best loss
        if (this->bestLoss < std::numeric_limits<T>::max()) {
          T threshold = static_cast<T>(config.metrics.lossExplosion.threshold);

          if (loss > threshold * this->bestLoss) {
            std::ostringstream oss;
            oss << "Loss explosion: current loss (" << loss << ") exceeds " << config.metrics.lossExplosion.threshold
                << "x best loss (" << this->bestLoss << ")";
            this->stopReason = oss.str();
            return true;
          }
        }

        return false;
      }
  };
}

//===================================================================================================================//

#endif // COMMON_TRAININGMONITOR_HPP
