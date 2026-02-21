#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_Loader.hpp"

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
    Runner(const QCommandLineParser& parser, bool verbose);
    int run();

  private:
    // ANN mode methods
    int runANNTrain();
    int runANNTest();
    int runANNPredict();

    // CNN mode methods
    int runCNNTrain();
    int runCNNTest();
    int runCNNPredict();

    // Load samples from JSON or IDX based on CLI options
    std::pair<ANN::Samples<float>, bool> loadANNSamplesFromOptions(
      const std::string& modeName, QString& inputFilePath);
    std::pair<CNN::Samples<float>, bool> loadCNNSamplesFromOptions(
      const std::string& modeName, QString& inputFilePath);

    // Output path helpers
    static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);
    static std::string generateDefaultOutputPath(
      const QString& inputFilePath, ulong epochs, ulong samples, float loss);

    const QCommandLineParser& parser_;
    bool verbose_;
    NetworkType networkType_;
    std::string mode_;  // "train", "test", "predict"

    // ANN members (used when networkType_ == ANN)
    std::unique_ptr<ANN::Core<float>> annCore_;
    ANN::CoreConfig<float> annCoreConfig_;

    // CNN members (used when networkType_ == CNN)
    std::unique_ptr<CNN::Core<float>> cnnCore_;
    CNN::CoreConfig<float> cnnCoreConfig_;
};

} // namespace NN_CLI

#endif // NN_CLI_RUNNER_HPP

