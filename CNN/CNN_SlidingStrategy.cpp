#include "CNN_SlidingStrategy.hpp"

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

SlidingStrategyType SlidingStrategy::nameToType(const std::string& name)
{
  auto it = slidingStrategyMap.find(name);

  if (it != slidingStrategyMap.end()) {
    return it->second;
  }

  throw std::runtime_error("Unknown sliding strategy: " + name);
}

//===================================================================================================================//

std::string SlidingStrategy::typeToName(const SlidingStrategyType& type)
{
  for (const auto& pair : slidingStrategyMap) {
    if (pair.second == type) {
      return pair.first;
    }
  }

  throw std::runtime_error("Unknown sliding strategy enum value");
}

//===================================================================================================================//

ulong SlidingStrategy::computePadding(ulong kernelSize, SlidingStrategyType strategy)
{
  switch (strategy) {
  case SlidingStrategyType::VALID:
    return 0;
  case SlidingStrategyType::FULL:
    return kernelSize - 1;
  case SlidingStrategyType::SAME:
    // SAME padding requires odd kernel sizes. With symmetric padding (pad = k/2
    // on each side), even kernels produce output LARGER than input:
    //   outH = (input + 2*(k/2) - k)/stride + 1 = (input)/stride + 1  (even k)
    // vs. the expected outH = ceil(input/stride).
    // For odd kernels: outH = (input + 2*(k/2) - k)/stride + 1 = input/stride + 1
    // (when stride=1) which equals input — correct SAME behavior.
    if (kernelSize % 2 == 0)
      throw std::runtime_error("SAME padding requires an odd kernel size (got " + std::to_string(kernelSize) +
                               "). Use VALID or "
                               "an odd kernel (3, 5, 7, ...).");
    return kernelSize / 2;
  default:
    throw std::runtime_error("Cannot compute padding for unknown sliding strategy");
  }
}
