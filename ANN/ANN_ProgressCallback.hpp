#ifndef ANN_PROGRESSCALLBACK_HPP
#define ANN_PROGRESSCALLBACK_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace ANN
{
  // Generic progress callback for test and predict modes.
  // Parameters: (currentSample, totalSamples)
  using ProgressCallback = std::function<void(ulong, ulong)>;
}

//===================================================================================================================//

#endif // ANN_PROGRESSCALLBACK_HPP
