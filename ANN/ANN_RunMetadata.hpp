#ifndef ANN_RUNMETADATA_HPP
#define ANN_RUNMETADATA_HPP

#include <string>

//===================================================================================================================//

namespace ANN {
  // Run metadata (captured at runtime when running inference)
  template <typename T>
  struct RunMetadata {
    std::string startTime;      // ISO 8601 format (e.g., "2026-02-19T10:30:00")
    std::string endTime;        // ISO 8601 format
    double durationSeconds;     // Total run duration in seconds
    std::string durationFormatted; // Human-readable duration (e.g., "0s", "1m 30s")
  };
}

//===================================================================================================================//

#endif // ANN_RUNMETADATA_HPP

