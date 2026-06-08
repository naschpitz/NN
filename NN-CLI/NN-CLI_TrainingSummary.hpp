#ifndef NN_CLI_TRAININGSUMMARY_HPP
#define NN_CLI_TRAININGSUMMARY_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_Types.hpp"

#include <_CoreConfig.hpp>
#include <CNN_CoreConfig.hpp>

#include <string>
#include <vector>

namespace NN_CLI
{

  class TrainingSummary
  {
    public:
      // Return table lines without printing (for ncurses rendering).
      static std::vector<std::string> collectCNN(const CNN::CoreConfig<float>& cnnConfig,
                                                 const AugmentationConfig& augConfig, ulong numOriginalTrainSamples,
                                                 ulong numTrainSamples, ulong numValidationSamples,
                                                 float validationRatio, bool validationAuto, ulong maxWidth = 0);

      static std::vector<SummaryRow> collectCNNRows(const CNN::CoreConfig<float>& cnnConfig,
                                                    const AugmentationConfig& augConfig, ulong numOriginalTrainSamples,
                                                    ulong numTrainSamples, ulong numValidationSamples,
                                                    float validationRatio, bool validationAuto);

      static std::vector<std::string> collect(const ANN::CoreConfig<float>& annConfig,
                                                 const AugmentationConfig& augConfig, ulong numOriginalTrainSamples,
                                                 ulong numTrainSamples, ulong numValidationSamples,
                                                 float validationRatio, bool validationAuto, ulong maxWidth = 0);

      static ulong countCNNParameters(const CNN::CoreConfig<float>& config);
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGSUMMARY_HPP
