#ifndef NN_CLI_TESTSUMMARY_HPP
#define NN_CLI_TESTSUMMARY_HPP

#include "NN-CLI_Types.hpp"

#include <_CoreConfig.hpp>
#include <CNN_CoreConfig.hpp>

namespace NN_CLI
{

  class TestSummary
  {
    public:
      static void printCNN(const CNN::CoreConfig<float>& cnnConfig, ulong testSamples);
      static void print(const ANN::CoreConfig<float>& annConfig, ulong testSamples);
  };

} // namespace NN_CLI

#endif // NN_CLI_TESTSUMMARY_HPP
