#ifndef ANN_PREDICTMETADATA_HPP
#define ANN_PREDICTMETADATA_HPP

#include <string>

//===================================================================================================================//

namespace ANN {
  // Predict metadata (captured at runtime when running predict)
  template <typename T>
  struct PredictMetadata {
    std::string startTime;      // ISO 8601 format (e.g., "2026-02-19T10:30:00")
    std::string endTime;        // ISO 8601 format
    double durationSeconds;     // Total predict duration in seconds
    std::string durationFormatted; // Human-readable duration (e.g., "0s", "1m 30s")
  };
}

//===================================================================================================================//

#endif // ANN_PREDICTMETADATA_HPP

