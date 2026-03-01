#ifndef ANN_SAMPLE_HPP
#define ANN_SAMPLE_HPP

#include "ANN_Types.hpp"

#include <functional>

//===================================================================================================================//

namespace ANN
{
  template <typename T>
  struct Sample {
      Input<T> input;
      Output<T> output;
  };

  template <typename T>
  using Samples = std::vector<Sample<T>>;

  // Callback that returns samples for a given batch.
  // Receives the full shuffled index array, batch size, and current batch index so the
  // provider can internally prefetch the next batch while the current one is being trained.
  template <typename T>
  using SampleProvider =
    std::function<Samples<T>(const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex)>;

  // Create a SampleProvider that serves from an in-memory Samples vector.
  // No prefetching needed — all data is already in memory.
  template <typename T>
  inline SampleProvider<T> makeSampleProvider(const Samples<T>& samples)
  {
    return [&samples](const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex) -> Samples<T> {
      ulong start = batchIndex * batchSize;
      ulong end = std::min(start + batchSize, static_cast<ulong>(sampleIndices.size()));

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

#endif // ANN_SAMPLE_HPP
