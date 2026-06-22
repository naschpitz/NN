#ifndef CNN_SAMPLEPROVIDER_HPP
#define CNN_SAMPLEPROVIDER_HPP

#include "CNN_Sample.hpp"
#include "Common/Common_Device.hpp"

#include <algorithm>
#include <functional>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Lazy supplier used by train() and test(): given the full shuffled index
  // array, a fetch size, and an absolute start index, returns the corresponding
  // window of samples. Lets callers stream samples on demand (e.g. decode
  // images window-by-window) instead of holding the full Samples<T> vector
  // in memory. fetchSize is independent of the mini-batch boundary — it
  // controls only how many samples are materialized per call.
  template <typename T>
  using SampleProvider =
    std::function<Samples<T>(const std::vector<ulong>& sampleIndices, ulong fetchSize, ulong fetchStart)>;

  // Build a SampleProvider that serves from an in-memory Samples vector.
  // No prefetching needed — all data is already in memory.
  template <typename T>
  inline SampleProvider<T> makeSampleProvider(const Samples<T>& samples)
  {
    return [&samples](const std::vector<ulong>& sampleIndices, ulong fetchSize, ulong fetchStart) -> Samples<T> {
      ulong start = fetchStart;
      ulong end = std::min(start + fetchSize, static_cast<ulong>(sampleIndices.size()));

      Samples<T> batch;
      batch.reserve(end - start);

      for (ulong i = start; i < end; i++) {
        batch.push_back(samples[sampleIndices[i]]);
      }

      return batch;
    };
  }
}

//===================================================================================================================//

#endif // CNN_SAMPLEPROVIDER_HPP
