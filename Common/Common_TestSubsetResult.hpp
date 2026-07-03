#ifndef COMMON_TESTSUBSETRESULT_HPP
#define COMMON_TESTSUBSETRESULT_HPP

#include "Common_ConfusionMatrix.hpp"

#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  // Per-GPU-subset result returned by CoreGPUWorker::testSubset().
  // Carries loss plus raw confusion counts; the orchestrator merges subsets and
  // calls ConfusionMatrix::computeMetrics() once on the aggregate. numCorrect is
  // then read from the aggregate's diagonal (trace).
  template <typename T>
  struct TestSubsetResult {
      T loss = 0;
      ConfusionMatrix<T> confusion;
  };
}

//===================================================================================================================//

#endif // COMMON_TESTSUBSETRESULT_HPP
