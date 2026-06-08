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
    // Log-linear interpolation between anchor points to avoid discontinuities.
    // Anchor points (samples → validation ratio):
    //   100 → 20%,  1k → 15%,  10k → 10%,  100k → 5%,  1M → 1%
    struct Anchor {
        double logN;
        double ratio;
    };

    static const Anchor anchors[] = {
      {std::log(100.0), 0.20},    {std::log(1000.0), 0.15},    {std::log(10000.0), 0.10},
      {std::log(100000.0), 0.05}, {std::log(1000000.0), 0.01},
    };

    static const int numAnchors = sizeof(anchors) / sizeof(anchors[0]);

    double logN = std::log(static_cast<double>(std::max(totalSamples, 1UL)));

    // Clamp to anchor range
    if (logN <= anchors[0].logN)
      return static_cast<float>(anchors[0].ratio);

    if (logN >= anchors[numAnchors - 1].logN)
      return static_cast<float>(anchors[numAnchors - 1].ratio);

    // Find the two anchors that bracket logN and interpolate
    for (int i = 0; i < numAnchors - 1; i++) {
      if (logN <= anchors[i + 1].logN) {
        double t = (logN - anchors[i].logN) / (anchors[i + 1].logN - anchors[i].logN);
        double ratio = anchors[i].ratio + t * (anchors[i + 1].ratio - anchors[i].ratio);
        return static_cast<float>(ratio);
      }
    }

    return 0.01f;
  }

} // namespace NN_CLI
