#ifndef COMMON_PROGRESSCALLBACK_HPP
#define COMMON_PROGRESSCALLBACK_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  // Generic progress callback for test and predict modes.
  // Parameters: (currentSample, totalSamples)
  using ProgressCallback = std::function<void(ulong, ulong)>;
}

//===================================================================================================================//

#endif // COMMON_PROGRESSCALLBACK_HPP
