#ifndef NN_CLI_PREDICTSUMMARY_HPP
#define NN_CLI_PREDICTSUMMARY_HPP

#include "NN-CLI_Types.hpp"

#include <ANN_CoreConfig.hpp>
#include <CNN_CoreConfig.hpp>

#include <string>

namespace NN_CLI
{

  class PredictSummary
  {
    public:
      static void printCNN(const CNN::CoreConfig<float>& cnnConfig, ulong numInputs, const std::string& inputPath,
                           const std::string& outputPath);
      static void print(const ANN::CoreConfig<float>& annConfig, ulong numInputs, const std::string& inputPath,
                           const std::string& outputPath);
  };

} // namespace NN_CLI

#endif // NN_CLI_PREDICTSUMMARY_HPP
