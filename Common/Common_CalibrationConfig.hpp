#ifndef COMMON_CALIBRATIONCONFIG_HPP
#define COMMON_CALIBRATIONCONFIG_HPP

#include "Common/Common_LogLevel.hpp"

#include <cstddef>
#include <string>

//===================================================================================================================//

namespace Common
{
  /**
   * Configuration for the calibration pipeline.
   *
   * Controls which image directories are used, how many samples are drawn,
   * which ID percentile becomes the OOD threshold, and where the result
   * JSON is written.
   */
  struct CalibrationConfig {
    std::string idImagesDir;
    std::string oodDir;
    std::size_t idSampleCount = 500;
    std::size_t oodSampleCount = 1500;
    double idPercentile = 95.0;
    std::string outputPath;
    bool fetchIfMissing = true;
    LogLevel logLevel = LogLevel::ERROR;
    ulong progressReports = 0;
  };
}

//===================================================================================================================//

#endif // COMMON_CALIBRATIONCONFIG_HPP
