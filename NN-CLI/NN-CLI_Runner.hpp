#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_Loader.hpp"
#include "NN-CLI_NetworkType.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <QCommandLineParser>

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  /**
 * Runner class handles the execution of ANN and CNN modes (train, test, predict).
 * Automatically detects network type from the config file and delegates to the
 * appropriate library.
 */
  class Runner
  {
    public:
      //-- Constructor --//
      Runner(const QCommandLineParser& parser, LogLevel logLevel);

      //-- Entry point --//
      int run();

    private:
      //-- ANN mode methods --//
      int runANNTrain();
      int runANNTest();
      int runANNPredict();

      //-- CNN mode methods --//
      int runCNNTrain();
      int runCNNTest();
      int runCNNPredict();

      //-- Sample loading --//
      std::pair<ANN::Samples<float>, bool> loadANNSamplesFromOptions(const std::string& modeName,
                                                                     QString& inputFilePath);
      std::pair<CNN::Samples<float>, bool> loadCNNSamplesFromOptions(const std::string& modeName,
                                                                     QString& inputFilePath);

      //-- Model saving --//
      void saveANNModel(const std::string& filePath) const;
      void saveCNNModel(const std::string& filePath) const;

      //-- Output path helpers --//
      static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);
      static std::string generateDefaultOutputPath(const QString& inputFilePath, ulong epochs, ulong samples,
                                                   float loss);
      static std::string generateCheckpointPath(const QString& inputFilePath, ulong epoch, float loss);

      //-- Training helpers --//
      void setupANNTrainingCallback(const QString& inputFilePath);
      void setupCNNTrainingCallback(const QString& inputFilePath);
      int finishANNTraining(const QString& inputFilePath);
      int finishCNNTraining(const QString& inputFilePath);

      //-- Class weight computation --//
      std::vector<float> computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs);

      //-- Configuration --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      NetworkType networkType;
      std::string mode; // "train", "test", "predict"
      IOConfig ioConfig; // inputType / outputType / shapes / progress / checkpoints (NN-CLI concept only)

      //-- Data augmentation config (parsed from trainingConfig, handled by NN-CLI only) --//
      Loader::AugmentationConfig augConfig;

      //-- ANN members --//
      std::unique_ptr<ANN::Core<float>> annCore;
      ANN::CoreConfig<float> annCoreConfig;

      //-- CNN members --//
      std::unique_ptr<CNN::Core<float>> cnnCore;
      CNN::CoreConfig<float> cnnCoreConfig;
  };

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP
