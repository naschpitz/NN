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
          bestLoss_(std::numeric_limits<T>::max()),
          bestEpoch_(0),
          epochsWithoutImprovement(0),
          newBest_(false),
          stopReason_()
      {
      }

      // Called at end of each epoch. Returns true if training should stop.
      bool checkEpoch(ulong epoch, T epochLoss, std::optional<T> validationLoss = std::nullopt)
      {
        newBest_ = false;

        // Use validation loss when available, otherwise training loss
        T loss = validationLoss.has_value() ? validationLoss.value() : epochLoss;

        // Update best loss tracking
        T delta = static_cast<T>(config.metrics.lossStagnation.minDelta);

        if (loss < bestLoss_ - delta) {
          bestLoss_ = loss;
          bestEpoch_ = epoch;
          newBest_ = true;
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
        if (!newBest_) {
          if (checkLossStagnation(loss)) {
            return true;
          }
        }

        return false;
      }

      //-- Accessors --//
      bool isNewBest() const { return newBest_; }
      std::string stopReason() const { return stopReason_; }
      ulong bestEpoch() const { return bestEpoch_; }
      T bestLoss() const { return bestLoss_; }

    private:
      //-- Configuration --//
      MonitoringConfig config;

      //-- Loss tracking --//
      T bestLoss_;
      ulong bestEpoch_;
      ulong epochsWithoutImprovement;
      bool newBest_;
      std::string stopReason_;

      //-- Metric evaluators --//
      bool checkLossStagnation(T loss)
      {
        if (!config.metrics.lossStagnation.enabled) {
          return false;
        }

        epochsWithoutImprovement++;

        if (epochsWithoutImprovement >= config.patience) {
          std::ostringstream oss;
          oss << "Loss stagnation: no improvement > " << config.metrics.lossStagnation.minDelta << " for " << config.patience
              << " check intervals (" << config.patience * config.checkInterval << " epochs)";
          stopReason_ = oss.str();
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
          stopReason_ = "Loss explosion: NaN detected";
          return true;
        }

        // Only check threshold if we have a valid best loss
        if (bestLoss_ < std::numeric_limits<T>::max()) {
          T threshold = static_cast<T>(config.metrics.lossExplosion.threshold);

          if (loss > threshold * bestLoss_) {
            std::ostringstream oss;
            oss << "Loss explosion: current loss (" << loss << ") exceeds " << config.metrics.lossExplosion.threshold
                << "x best loss (" << bestLoss_ << ")";
            stopReason_ = oss.str();
            return true;
          }
        }

        return false;
      }
  };
}

//===================================================================================================================//

#endif // COMMON_TRAININGMONITOR_HPP
