#ifndef COMMON_EPOCHRECORD_HPP
#define COMMON_EPOCHRECORD_HPP

#include <cstdint>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  // Per-epoch record for tracking training history across epochs
  template <typename T>
  struct EpochRecord
  {
      //-- Members --//
      ulong    epoch          = 0;  // 0-based epoch index
      T        loss           = 0;  // training loss
      T        valLoss        = 0;  // validation loss (only meaningful if hasValLoss is true)
      bool     hasValLoss     = false; // whether validation ran this epoch
      bool     isBest         = false; // whether this epoch produced the best model so far
      uint64_t completionTime = 0;  // epoch completion time as unix timestamp (seconds since epoch)
  };
}

//===================================================================================================================//

#endif // COMMON_EPOCHRECORD_HPP
