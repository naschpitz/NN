#ifndef NN_CLI_ANNRUNNER_HPP
#define NN_CLI_ANNRUNNER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI.hpp"
#include "NN-CLI_TrainingTui.hpp"
#include "NN-CLI_Utils.hpp"

#include <ANN_Core.hpp>
#include "Common/Common_TrainingMonitor.hpp"

#include <QCommandLineParser>

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
      void setupTrainingCallback(const QString& inputFilePath,
                                 std::shared_ptr<ANN::Core<float>> validationCore = nullptr,
                                 std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<ANN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);
      int finishTraining(const QString& inputFilePath);
      ValidationMetadata buildValidationMetadata() const;

      //-- References to shared state (owned by Runner) --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;
      std::unique_ptr<ANN::Core<float>>& core;
      ANN::CoreConfig<float>& coreConfig;

      //-- Validation state --//
      ValidationState validationState;

      //-- Callback State --//
      ulong lastCallbackEpoch = 0;
      float lastEpochLoss = 0.0f;
      bool lastIsBest = false;
      bool lastHadValLoss = false;
      float lastValLoss = 0.0f;
      bool cacheIsSet = false;
      std::mutex epochTransitionMutex;
      std::unique_ptr<ProgressBar> progressBar;

      //-- Validation objects (stored during train() for finishTraining()) --//
      std::shared_ptr<ANN::Core<float>> validationCore;
      std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;
      std::shared_ptr<std::vector<ulong>> validationIndices;
      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Loading-bar wiring shared with the CNN runner --//
      TrainingTui trainingTui;
  };

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP
