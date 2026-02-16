#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDebug>

#include <ANN_Core.hpp>
#include <ANN_CoreMode.hpp>
#include <ANN_CoreType.hpp>
#include <ANN_Utils.hpp>

#include "ANN-CLI_Loader.hpp"
#include "ANN-CLI_Utils.hpp"

#include <iostream>
#include <memory>

void printUsage() {
  std::cout << "ANN-CLI - Artificial Neural Network Command Line Interface\n\n";
  std::cout << "Usage:\n";
  std::cout << "  ANN-CLI --config <file> --mode <train|run> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config, -c <file>    Path to JSON configuration file\n";
  std::cout << "  --mode, -m <mode>      Mode: 'train' or 'run'\n";
  std::cout << "  --type, -t <type>      Core type: 'cpu' or 'gpu' (default: cpu)\n";
  std::cout << "  --input, -i <file>     Path to JSON file with input values for run mode\n";
  std::cout << "  --samples, -s <file>   Path to JSON file with training samples\n";
  std::cout << "  --idx-data <file>      Path to IDX3 data file (alternative to --samples)\n";
  std::cout << "  --idx-labels <file>    Path to IDX1 labels file (requires --idx-data)\n";
  std::cout << "  --output, -o <file>    Output file for saving trained model\n";
  std::cout << "  --help, -h             Show this help message\n";
}

int main(int argc, char *argv[]) {
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("ANN-CLI");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Artificial Neural Network CLI");
  parser.addHelpOption();
  parser.addVersionOption();

  // Config file option
  QCommandLineOption configOption(
    QStringList() << "c" << "config",
    "Path to JSON configuration file.",
    "file"
  );
  parser.addOption(configOption);

  // Mode option (train or run)
  QCommandLineOption modeOption(
    QStringList() << "m" << "mode",
    "Mode: 'train' or 'run'.",
    "mode"
  );
  parser.addOption(modeOption);

  // Core type option (cpu or gpu)
  QCommandLineOption typeOption(
    QStringList() << "t" << "type",
    "Core type: 'cpu' or 'gpu' (default: cpu).",
    "type",
    "cpu"
  );
  parser.addOption(typeOption);

  // Input file for run mode
  QCommandLineOption inputOption(
    QStringList() << "i" << "input",
    "Path to JSON file with input values for run mode.",
    "file"
  );
  parser.addOption(inputOption);

  // Samples file for training (JSON format)
  QCommandLineOption samplesOption(
    QStringList() << "s" << "samples",
    "Path to JSON file with training samples.",
    "file"
  );
  parser.addOption(samplesOption);

  // IDX data file for training (IDX3 format)
  QCommandLineOption idxDataOption(
    QStringList() << "idx-data",
    "Path to IDX3 data file (alternative to --samples).",
    "file"
  );
  parser.addOption(idxDataOption);

  // IDX labels file for training (IDX1 format)
  QCommandLineOption idxLabelsOption(
    QStringList() << "idx-labels",
    "Path to IDX1 labels file (requires --idx-data).",
    "file"
  );
  parser.addOption(idxLabelsOption);

  // Output file for saving trained model
  QCommandLineOption outputOption(
    QStringList() << "o" << "output",
    "Output file for saving trained model.",
    "file"
  );
  parser.addOption(outputOption);

  parser.process(app);

  // Validate required options
  if (!parser.isSet(configOption)) {
    std::cerr << "Error: --config option is required.\n\n";
    printUsage();
    return 1;
  }

  if (!parser.isSet(modeOption)) {
    std::cerr << "Error: --mode option is required.\n\n";
    printUsage();
    return 1;
  }

  QString configPath = parser.value(configOption);
  QString mode = parser.value(modeOption).toLower();
  QString coreTypeStr = parser.value(typeOption).toLower();

  if (mode != "train" && mode != "run") {
    std::cerr << "Error: Mode must be 'train' or 'run'.\n";
    return 1;
  }

  if (coreTypeStr != "cpu" && coreTypeStr != "gpu") {
    std::cerr << "Error: Type must be 'cpu' or 'gpu'.\n";
    return 1;
  }

  // Convert mode string to enum
  ANN::CoreModeType modeType = (mode == "train") ? ANN::CoreModeType::TRAIN : ANN::CoreModeType::RUN;

  // Convert core type string to enum
  ANN::CoreTypeType coreType = (coreTypeStr == "gpu") ? ANN::CoreTypeType::GPU : ANN::CoreTypeType::CPU;

  try {
    // Load the ANN configuration from JSON file
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << mode.toStdString() << ", Core type: " << coreTypeStr.toStdString() << "\n";

    ANN::CoreConfig<float> coreConfig = ANN_CLI::Loader::loadConfig(configPath.toStdString(), modeType, coreType);
    auto core = ANN::Core<float>::makeCore(coreConfig);

    if (mode == "train") {
      // Training mode
      ANN::Samples<float> samples;

      bool hasJsonSamples = parser.isSet(samplesOption);
      bool hasIdxData = parser.isSet(idxDataOption);
      bool hasIdxLabels = parser.isSet(idxLabelsOption);

      if (hasJsonSamples && hasIdxData) {
        std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
        return 1;
      }

      if (hasJsonSamples) {
        // Load from JSON format
        QString samplesPath = parser.value(samplesOption);
        std::cout << "Loading training samples from JSON: " << samplesPath.toStdString() << "\n";
        samples = ANN_CLI::Loader::loadSamples(samplesPath.toStdString());
      } else if (hasIdxData) {
        // Load from IDX format
        if (!hasIdxLabels) {
          std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
          return 1;
        }

        QString idxDataPath = parser.value(idxDataOption);
        QString idxLabelsPath = parser.value(idxLabelsOption);

        std::cout << "Loading training samples from IDX:\n";
        std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
        std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";

        samples = ANN_CLI::Utils<float>::loadIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
      } else {
        std::cerr << "Error: Training requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
        return 1;
      }

      std::cout << "Loaded " << samples.size() << " training samples.\n";

      std::cout << "Starting training...\n";
      core->train(samples);
      std::cout << "Training completed.\n";

      // Save the trained model if output file specified
      if (parser.isSet(outputOption)) {
        QString outputPath = parser.value(outputOption);
        ANN::Utils<float>::save(*core, outputPath.toStdString());
        std::cout << "Model saved to: " << outputPath.toStdString() << "\n";
      }

    } else if (mode == "run") {
      // Run mode
      if (!parser.isSet(inputOption)) {
        std::cerr << "Error: --input option is required for run mode.\n";
        return 1;
      }

      QString inputPath = parser.value(inputOption);
      std::cout << "Loading input from: " << inputPath.toStdString() << "\n";

      ANN::Input<float> input = ANN_CLI::Loader::loadInput(inputPath.toStdString());
      std::cout << "Running ANN with input: ";

      for (size_t i = 0; i < input.size(); ++i) {
        std::cout << input[i];
        if (i < input.size() - 1) std::cout << ", ";
      }

      std::cout << "\n";

      ANN::Output<float> output = core->run(input);

      std::cout << "Output: ";

      for (size_t i = 0; i < output.size(); ++i) {
        std::cout << output[i];
        if (i < output.size() - 1) std::cout << ", ";
      }

      std::cout << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
