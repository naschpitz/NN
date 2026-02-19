#ifndef ANN_TESTRESULT_HPP
#define ANN_TESTRESULT_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN {
  // Test result information
  template <typename T>
  struct TestResult {
    ulong numSamples;     // Total number of samples tested
    T totalLoss;          // Sum of all sample losses
    T averageLoss;        // Average loss per sample (totalLoss / numSamples)
  };
}

//===================================================================================================================//

#endif // ANN_TESTRESULT_HPP

