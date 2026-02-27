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

namespace NN_CLI {

/**
 * Runner class handles the execution of ANN and CNN modes (train, test, predict).
 * Automatically detects network type from the config file and delegates to the
 * appropriate library.
 */
class Runner {
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
    std::pair<ANN::Samples<float>, bool> loadANNSamplesFromOptions(
      const std::string& modeName, QString& inputFilePath);
    std::pair<CNN::Samples<float>, bool> loadCNNSamplesFromOptions(
      const std::string& modeName, QString& inputFilePath);

    //-- Model saving --//
    static void saveANNModel(const ANN::Core<float>& core, const std::string& filePath,
                              const IOConfig& ioConfig, ulong progressReports, ulong saveModelInterval);
    static void saveCNNModel(const CNN::Core<float>& core, const std::string& filePath,
                              const IOConfig& ioConfig, ulong progressReports, ulong saveModelInterval);

    //-- Output path helpers --//
    static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);
    static std::string generateDefaultOutputPath(
      const QString& inputFilePath, ulong epochs, ulong samples, float loss);
    static std::string generateCheckpointPath(
      const QString& inputFilePath, ulong epoch, float loss);

    //-- Data augmentation helpers --//
    template <typename SampleT>
    void augmentSamples(std::vector<SampleT>& samples);

    template <typename SampleT>
    std::vector<float> computeClassWeights(const std::vector<SampleT>& samples);

    //-- Configuration --//
    const QCommandLineParser& parser;
    LogLevel logLevel;
    NetworkType networkType;
    std::string mode;  // "train", "test", "predict"
    IOConfig ioConfig;  // inputType / outputType / shapes (NN-CLI concept only)
    ulong progressReports = 1000;  // NN-CLI display frequency (not used by ANN/CNN libs)
    ulong saveModelInterval = 10;  // 0 = disabled

    //-- Data augmentation config (parsed from trainingConfig, handled by NN-CLI only) --//
    ulong augmentationFactor = 0;       // 0 = disabled; N = N× total samples per class
    bool balanceAugmentation = false;   // true = augment minority classes up to max class count
    bool autoClassWeights = false;      // true = auto-compute inverse-frequency class weights
    float augmentationProbability = 0.5f; // Probability of applying each enabled transform
    Loader::AugmentationTransforms augTransforms; // Which transforms to apply

    //-- ANN members --//
    std::unique_ptr<ANN::Core<float>> annCore;
    ANN::CoreConfig<float> annCoreConfig;

    //-- CNN members --//
    std::unique_ptr<CNN::Core<float>> cnnCore;
    CNN::CoreConfig<float> cnnCoreConfig;
};

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP

