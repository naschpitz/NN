#ifndef NN_CLI_TRAININGSUMMARY_HPP
#define NN_CLI_TRAININGSUMMARY_HPP

#include "NN-CLI_AugmentationConfig.hpp"

#include <ANN_CoreConfig.hpp>
#include <CNN_CoreConfig.hpp>

#include <string>
#include <vector>

namespace NN_CLI
{

  using ulong = unsigned long;

  class TrainingSummary
  {
    public:
      // Print a formatted training configuration table to stdout.
      static void printCNN(const CNN::CoreConfig<float>& cnnConfig, const AugmentationConfig& augConfig,
                           ulong numOriginalTrainSamples, ulong numTrainSamples, ulong numValidationSamples,
                           float validationRatio, bool validationAuto);

      static void printANN(const ANN::CoreConfig<float>& annConfig, const AugmentationConfig& augConfig,
                           ulong numOriginalTrainSamples, ulong numTrainSamples, ulong numValidationSamples,
                           float validationRatio, bool validationAuto);

      // Count total trainable parameters from config (no model instantiation needed).
      static ulong countCNNParameters(const CNN::CoreConfig<float>& config);
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGSUMMARY_HPP
