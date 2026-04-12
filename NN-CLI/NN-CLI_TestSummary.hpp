#ifndef NN_CLI_TESTSUMMARY_HPP
#define NN_CLI_TESTSUMMARY_HPP

#include <ANN_CoreConfig.hpp>
#include <CNN_CoreConfig.hpp>

namespace NN_CLI
{

  using ulong = unsigned long;

  class TestSummary
  {
    public:
      static void printCNN(const CNN::CoreConfig<float>& cnnConfig, ulong testSamples);
      static void printANN(const ANN::CoreConfig<float>& annConfig, ulong testSamples);
  };

} // namespace NN_CLI

#endif // NN_CLI_TESTSUMMARY_HPP
