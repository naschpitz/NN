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
#include <ANN_TrainingMonitor.hpp>

#include <QCommandLineParser>

#include <limits>
#include <memory>
#include <mutex>
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
      void setupTrainingCallback(const QString& inputFilePath,
                                 std::shared_ptr<ANN::Core<float>> validationCore = nullptr,
                                 std::shared_ptr<ANN::TrainingMonitor<float>> trainingMonitor = nullptr,
                                 const DataLoader<ANN::Sample<float>>* validationDataLoader = nullptr,
                                 const std::vector<ulong>* validationIndices = nullptr);
      int finishTraining(const QString& inputFilePath);
      ValidationMetadata buildValidationMetadata() const;
      void regenerateConfigLines(ulong maxWidth);

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
      ulong lastCallbackEpoch_ = 0;
      float lastEpochLoss_ = 0.0f;
      std::mutex epochTransitionMutex_;
      std::unique_ptr<ProgressBar> progressBar_;

      //-- ncurses terminal UI (only active during training) --//
      std::shared_ptr<TerminalUI> tui;

      //-- Loading-bar wiring shared with the CNN runner --//
      TrainingTui trainingTui_;

      //-- Cached config for on-resize regeneration --//
      ulong cachedNumOrigTrainSamples_ = 0;
      ulong cachedNumTrainSamples_ = 0;
      ulong cachedNumValSamples_ = 0;
      float cachedValRatio_ = 0.0f;
      bool cachedValAuto_ = false;
      ulong cachedNumOutputClasses_ = 0;
      bool configLinesLoaded_ = false;
  };

} // namespace NN_CLI

#endif // NN_CLI_ANNRUNNER_HPP
