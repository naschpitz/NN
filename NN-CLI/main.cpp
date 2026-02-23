#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>

#include "NN-CLI_Runner.hpp"
#include "NN-CLI_LogLevel.hpp"

#include <iostream>
#include <string>

void printUsage() {
  std::cout << "NN-CLI - Neural Network Command Line Interface (ANN + CNN)\n\n";
  std::cout << "Usage:\n";
  std::cout << "  NN-CLI --config <file> --mode train [options]       # Training\n";
  std::cout << "  NN-CLI --config <file> --mode predict --input <f>   # Predict (batch)\n";
  std::cout << "  NN-CLI --config <file> --mode test [options]        # Evaluation\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config, -c <file>    Path to JSON configuration file (required)\n";
  std::cout << "  --mode, -m <mode>      Mode: 'train', 'predict', or 'test' (overrides config file)\n";
  std::cout << "  --device, -d <device>  Device: 'cpu' or 'gpu' (overrides config file)\n";
  std::cout << "  --input, -i <file>     Path to JSON file with batch inputs (predict mode, required)\n";
  std::cout << "  --input-type <type>    Input data type: 'vector' or 'image' (overrides config file)\n";
  std::cout << "  --samples, -s <file>   Path to JSON file with samples (train/test modes)\n";
  std::cout << "  --idx-data <file>      Path to IDX3 data file (alternative to --samples)\n";
  std::cout << "  --idx-labels <file>    Path to IDX1 labels file (requires --idx-data)\n";
  std::cout << "  --output, -o <file>    Output file/dir (default: predict_<input>.json or folder for images)\n";
  std::cout << "  --output-type <type>   Output data type: 'vector' or 'image' (overrides config file)\n";
  std::cout << "  --log-level, -l <lvl>  Log level: quiet, error, warning, info, debug (default: error)\n";
  std::cout << "  --help, -h             Show this help message\n";
}

int main(int argc, char *argv[]) {
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("NN-CLI");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Neural Network CLI (ANN + CNN)");
  parser.addHelpOption();

  // Config file option
  QCommandLineOption configOption(
    QStringList() << "c" << "config",
    "Path to JSON configuration file.",
    "file"
  );
  parser.addOption(configOption);

  // Mode option (train, predict, or test)
  QCommandLineOption modeOption(
    QStringList() << "m" << "mode",
    "Mode: 'train', 'predict', or 'test'.",
    "mode"
  );
  parser.addOption(modeOption);

  // Device option (cpu or gpu)
  QCommandLineOption deviceOption(
    QStringList() << "d" << "device",
    "Device: 'cpu' or 'gpu' (default: cpu).",
    "device",
    "cpu"
  );
  parser.addOption(deviceOption);

  // Input file for predict mode
  QCommandLineOption inputOption(
    QStringList() << "i" << "input",
    "Path to JSON file with input values for predict mode.",
    "file"
  );
  parser.addOption(inputOption);

  // Input type option (vector or image)
  QCommandLineOption inputTypeOption(
    QStringList() << "input-type",
    "Input data type: 'vector' or 'image' (overrides config file).",
    "type"
  );
  parser.addOption(inputTypeOption);

  // Samples file for training/testing (JSON format)
  QCommandLineOption samplesOption(
    QStringList() << "s" << "samples",
    "Path to JSON file with samples (for train/test modes).",
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

  // Output file (train: model, predict: predict result with metadata)
  QCommandLineOption outputOption(
    QStringList() << "o" << "output",
    "Output file. Train mode: saves trained model. Predict mode: saves predict result with model metadata.",
    "file"
  );
  parser.addOption(outputOption);

  // Output type option (vector or image)
  QCommandLineOption outputTypeOption(
    QStringList() << "output-type",
    "Output data type: 'vector' or 'image' (overrides config file).",
    "type"
  );
  parser.addOption(outputTypeOption);

  // Log level option
  QCommandLineOption logLevelOption(
    QStringList() << "l" << "log-level",
    "Log level: quiet, error, warning, info, debug (default: error).",
    "level",
    "error"
  );
  parser.addOption(logLevelOption);

  parser.process(app);

  // Validate that --config is provided
  if (!parser.isSet(configOption)) {
    std::cerr << "Error: --config is required.\n\n";
    printUsage();
    return 1;
  }

  // Validate mode if provided
  if (parser.isSet(modeOption)) {
    QString modeStr = parser.value(modeOption).toLower();
    if (modeStr != "train" && modeStr != "predict" && modeStr != "test") {
      std::cerr << "Error: Mode must be 'train', 'predict', or 'test'.\n";
      return 1;
    }
  }

  // Validate device if provided
  if (parser.isSet(deviceOption)) {
    QString deviceStr = parser.value(deviceOption).toLower();
    if (deviceStr != "cpu" && deviceStr != "gpu") {
      std::cerr << "Error: Device must be 'cpu' or 'gpu'.\n";
      return 1;
    }
  }

  // Validate input-type if provided
  if (parser.isSet(inputTypeOption)) {
    QString typeStr = parser.value(inputTypeOption).toLower();
    if (typeStr != "vector" && typeStr != "image") {
      std::cerr << "Error: Input type must be 'vector' or 'image'.\n";
      return 1;
    }
  }

  // Validate output-type if provided
  if (parser.isSet(outputTypeOption)) {
    QString typeStr = parser.value(outputTypeOption).toLower();
    if (typeStr != "vector" && typeStr != "image") {
      std::cerr << "Error: Output type must be 'vector' or 'image'.\n";
      return 1;
    }
  }

  // Parse log level
  NN_CLI::LogLevel logLevel = NN_CLI::LogLevel::ERROR;
  if (parser.isSet(logLevelOption)) {
    QString levelStr = parser.value(logLevelOption).toLower();
    if (levelStr == "quiet")        logLevel = NN_CLI::LogLevel::QUIET;
    else if (levelStr == "error")   logLevel = NN_CLI::LogLevel::ERROR;
    else if (levelStr == "warning") logLevel = NN_CLI::LogLevel::WARNING;
    else if (levelStr == "info")    logLevel = NN_CLI::LogLevel::INFO;
    else if (levelStr == "debug")   logLevel = NN_CLI::LogLevel::DEBUG;
    else {
      std::cerr << "Error: Log level must be 'quiet', 'error', 'warning', 'info', or 'debug'.\n";
      return 1;
    }
  }

  try {
    NN_CLI::Runner runner(parser, logLevel);
    return runner.run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
