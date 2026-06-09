#ifndef COMMON_TRAININGMETADATA_HPP
#define COMMON_TRAININGMETADATA_HPP

#include "Common/Common_EpochRecord.hpp"

#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace Common
{
  // Training metadata (captured at runtime, saved with the model)
  template <typename T>
  struct TrainingMetadata {
      std::string startTime; // ISO 8601 format (e.g., "2026-02-19T10:30:00")
      std::string endTime; // ISO 8601 format
      double durationSeconds; // Total training duration in seconds
      std::string durationFormatted; // Human-readable duration (e.g., "1y 2mo 3d 4h 5m 6s")
      ulong numSamples; // Number of training samples used
      T finalLoss; // Average loss at the end of training

      // Monitoring fields
      ulong lastEpoch = 0; // Epoch at which this model was saved
      std::string stopReason; // Why training stopped (empty = completed all epochs)
      ulong bestEpoch = 0; // Epoch with best loss
      T bestLoss = 0; // Best loss value

      // Per-epoch training history
      std::vector<EpochRecord<T>> epochHistory;
  };
}

//===================================================================================================================//

#endif // COMMON_TRAININGMETADATA_HPP
