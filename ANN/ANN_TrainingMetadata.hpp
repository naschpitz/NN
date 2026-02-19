#ifndef ANN_TRAININGMETADATA_HPP
#define ANN_TRAININGMETADATA_HPP

#include <string>
#include <sys/types.h>

//===================================================================================================================//

namespace ANN {
  // Training metadata (captured at runtime, saved with the model)
  template <typename T>
  struct TrainingMetadata {
    std::string startTime;      // ISO 8601 format (e.g., "2026-02-19T10:30:00")
    std::string endTime;        // ISO 8601 format
    double durationSeconds;     // Total training duration in seconds
    std::string durationFormatted; // Human-readable duration (e.g., "1y 2mo 3d 4h 5m 6s")
    ulong numSamples;           // Number of training samples used
    T finalLoss;                // Average loss at the end of training
  };
}

//===================================================================================================================//

#endif // ANN_TRAININGMETADATA_HPP

