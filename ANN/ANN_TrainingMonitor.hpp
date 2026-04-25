#ifndef ANN_TRAININGMONITOR_HPP
#define ANN_TRAININGMONITOR_HPP

#include "ANN_MonitoringConfig.hpp"

#include <optional>
#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  class TrainingMonitor
  {
    public:
      TrainingMonitor(const MonitoringConfig& config);

      // Called at end of each epoch. Returns true if training should stop.
      bool checkEpoch(ulong epoch, T epochLoss, std::optional<T> validationLoss = std::nullopt);

      //-- Accessors --//
      bool isNewBest() const;
      std::string stopReason() const;
      ulong bestEpoch() const;
      T bestLoss() const;

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
      bool checkLossStagnation(T loss);
      bool checkLossExplosion(T loss);
  };
}

//===================================================================================================================//

#endif // ANN_TRAININGMONITOR_HPP
