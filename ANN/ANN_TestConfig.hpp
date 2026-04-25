#ifndef ANN_TESTCONFIG_HPP
#define ANN_TESTCONFIG_HPP

#include <sys/types.h>

//===================================================================================================================//

namespace ANN
{
  struct TestConfig {
    ulong batchSize = 64; // Mini-batch size for test evaluation (default = 64)
  };
}

//===================================================================================================================//

#endif // ANN_TESTCONFIG_HPP
