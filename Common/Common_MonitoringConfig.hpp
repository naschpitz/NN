#ifndef COMMON_MONITORINGCONFIG_HPP
#define COMMON_MONITORINGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  struct LossStagnationConfig {
      bool enabled = true;
      float minDelta = 0.0001f;
  };

  struct LossExplosionConfig {
      bool enabled = true;
      float threshold = 10.0f;
  };

  struct MonitoringMetrics {
      LossStagnationConfig lossStagnation;
      LossExplosionConfig lossExplosion;
  };

  struct MonitoringConfig {
      bool enabled = false;
      ulong checkInterval = 5;
      ulong patience = 20;
      MonitoringMetrics metrics;
  };
}

//===================================================================================================================//

#endif // COMMON_MONITORINGCONFIG_HPP
