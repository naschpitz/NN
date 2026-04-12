#include "NN-CLI_DataSplitter.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>

namespace NN_CLI
{

  //===================================================================================================================//

  DataSplit DataSplitter::stratifiedSplit(const std::vector<std::vector<float>>& outputs, float ratio, ulong seed)
  {
    // Group sample indices by class (argmax of output vector)
    std::map<ulong, std::vector<ulong>> classIndices;

    for (ulong i = 0; i < outputs.size(); i++) {
      const auto& output = outputs[i];
      ulong cls = static_cast<ulong>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
      classIndices[cls].push_back(i);
    }

    DataSplit split;
    std::mt19937 rng(seed);

    for (auto& [cls, indices] : classIndices) {
      std::shuffle(indices.begin(), indices.end(), rng);
      ulong validationCount = static_cast<ulong>(std::round(indices.size() * ratio));

      for (ulong i = 0; i < indices.size(); i++) {
        if (i < validationCount)
          split.validationIndices.push_back(indices[i]);
        else
          split.trainIndices.push_back(indices[i]);
      }
    }

    return split;
  }

  //===================================================================================================================//

  float DataSplitter::computeAutoValSize(ulong totalSamples)
  {
    if (totalSamples > 100000)
      return 0.05f;

    if (totalSamples > 10000)
      return 0.10f;

    if (totalSamples > 1000)
      return 0.15f;

    return 0.20f;
  }

} // namespace NN_CLI
