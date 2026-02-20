#ifndef ANN_CLI_RUNNER_HPP
#define ANN_CLI_RUNNER_HPP

#include <ANN_Core.hpp>
#include <ANN_Mode.hpp>
#include <ANN_Device.hpp>

#include <QCommandLineParser>

#include <memory>
#include <optional>
#include <string>

//===================================================================================================================//

namespace ANN_CLI {

/**
 * Runner class handles the execution of different ANN modes (train, test, inference).
 * Encapsulates mode-specific logic that was previously in main().
 */
class Runner {
  public:
    /**
     * Construct a Runner with parsed command-line options.
     * @param parser The parsed command line parser
     * @param verbose Whether to print detailed information
     */
    Runner(const QCommandLineParser& parser, bool verbose);

    /**
     * Execute the appropriate mode based on configuration.
     * @return Exit code (0 for success, non-zero for error)
     */
    int run();

  private:
    /**
     * Execute training mode.
     * @return Exit code (0 for success, non-zero for error)
     */
    int runTrain();

    /**
     * Execute test/evaluation mode.
     * @return Exit code (0 for success, non-zero for error)
     */
    int runTest();

    /**
     * Execute inference mode.
     * @return Exit code (0 for success, non-zero for error)
     */
    int runInference();

    /**
     * Load samples from either JSON or IDX format based on CLI options.
     * @param modeName Name of the mode for error messages (e.g., "Training", "Test")
     * @param inputFilePath Output: set to the input file path used
     * @return Pair of (samples, success). If success is false, samples are empty.
     */
    std::pair<ANN::Samples<float>, bool> loadSamplesFromOptions(
      const std::string& modeName,
      QString& inputFilePath);

    /**
     * Generate default output filename with training info.
     */
    static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);

    /**
     * Generate default output path based on input file location.
     */
    static std::string generateDefaultOutputPath(
      const QString& inputFilePath,
      ulong epochs,
      ulong samples,
      float loss);

    const QCommandLineParser& parser_;
    bool verbose_;
    std::unique_ptr<ANN::Core<float>> core_;
    ANN::CoreConfig<float> coreConfig_;
};

} // namespace ANN_CLI

#endif // ANN_CLI_RUNNER_HPP

