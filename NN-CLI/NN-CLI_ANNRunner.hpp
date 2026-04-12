#ifndef NN_CLI_ANNRUNNER_HPP
#define NN_CLI_ANNRUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"

#include <ANN_Core.hpp>

#include <QCommandLineParser>

#include <limits>
#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  class ANNRunner
  {
    public:
      ANNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                std::unique_ptr<ANN::Core<float>>& core, ANN::CoreConfig<float>& coreConfig);

      int train();
      int test();
      int predict();

    private:
      //-- Sample loading --//
      std::pair<ANN::Samples<float>, bool> loadSamplesFromOptions(const std::string& modeName, QString& inputFilePath);

      //-- Training helpers --//
      void setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<ANN::Core<float>> valCore = nullptr,
                                 const DataLoader<ANN::Sample<float>>* valDataLoader = nullptr,
                                 const std::vector<ulong>* valIndices = nullptr);
      int finishTraining(const QString& inputFilePath);
      ValidationMetadata buildValidationMetadata() const;

      //-- Class weight computation --//
      static std::vector<float> computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs);

      //-- References to shared state (owned by Runner) --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<ANN::Core<float>>& core;
      ANN::CoreConfig<float>& coreConfig;

      //-- Validation state --//
      struct ValidationState {
          bool enabled = false;
          ulong checkInterval = 1;
          ulong numValSamples = 0;
          float bestValLoss = std::numeric_limits<float>::max();
          ulong bestValEpoch = 0;
          float lastValLoss = 0.0f;
      };

      ValidationState valState;
  };

} // namespace NN_CLI

#endif // NN_CLI_ANNRUNNER_HPP
