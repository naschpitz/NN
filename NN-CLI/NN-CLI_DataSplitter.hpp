#ifndef NN_CLI_DATASPLITTER_HPP
#define NN_CLI_DATASPLITTER_HPP

#include <sys/types.h>

#include <vector>

namespace NN_CLI
{

  using ulong = unsigned long;

  struct DataSplit {
      std::vector<ulong> trainIndices;
      std::vector<ulong> valIndices;
  };

  class DataSplitter
  {
    public:
      // Stratified split — preserves class distribution in both sets.
      // outputs: one output vector per sample (used to determine class via argmax).
      // ratio: fraction of samples to hold out for validation (0.0–1.0).
      // seed: RNG seed for deterministic, reproducible splits.
      static DataSplit stratifiedSplit(const std::vector<std::vector<float>>& outputs, float ratio, ulong seed = 42);

      // Auto-select validation ratio based on dataset size.
      static float computeAutoValSize(ulong totalSamples);
  };

} // namespace NN_CLI

#endif // NN_CLI_DATASPLITTER_HPP
