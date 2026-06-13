#ifndef COMMON_CALIBRATIONCONFIG_HPP
#define COMMON_CALIBRATIONCONFIG_HPP

#include <cstddef>
#include <string>

//===================================================================================================================//

namespace Common
{
  /**
    * Configuration for the calibration pipeline.
    *
    * Controls how many samples are drawn, which ID percentile becomes
    * the OOD threshold, and whether missing OOD datasets should be
    * auto-fetched.
    */
   struct CalibrationConfig {
     std::size_t idSampleCount = 500;
     std::size_t oodSampleCount = 1500;
     double idPercentile = 95.0;
     bool fetchIfMissing = true;
   };
}

//===================================================================================================================//

#endif // COMMON_CALIBRATIONCONFIG_HPP
