#ifndef COMMON_TESTCONFIG_HPP
#define COMMON_TESTCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN
{
  struct TestConfig {
      ulong batchSize = 64; // Mini-batch size for test evaluation (default = 64)
  };
}

//===================================================================================================================//

#endif // COMMON_TESTCONFIG_HPP
