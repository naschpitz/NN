#ifndef COMMON_TESTRESULT_HPP
#define COMMON_TESTRESULT_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  // Test result information
  template <typename T>
  struct TestResult {
      ulong numSamples; // Total number of samples tested
      T totalLoss; // Sum of all sample losses
      T averageLoss; // Average loss per sample (totalLoss / numSamples)
      ulong numCorrect; // Number of correctly classified samples (argmax match)
      T accuracy; // Percentage of correct classifications (0-100)
  };
}

//===================================================================================================================//

#endif // COMMON_TESTRESULT_HPP
