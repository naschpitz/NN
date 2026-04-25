#ifndef ANN_MONITORINGCONFIG_HPP
#define ANN_MONITORINGCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN
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

#endif // ANN_MONITORINGCONFIG_HPP
