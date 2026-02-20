#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>

#include "ANN-CLI_Runner.hpp"

#include <iostream>

void printUsage() {
  std::cout << "ANN-CLI - Artificial Neural Network Command Line Interface\n\n";
  std::cout << "Usage:\n";
  std::cout << "  ANN-CLI --config <file> --mode train [options]       # Training\n";
  std::cout << "  ANN-CLI --config <file> --mode inference --input <f> # Inference\n";
  std::cout << "  ANN-CLI --config <file> --mode test [options]        # Evaluation\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config, -c <file>    Path to JSON configuration file (required)\n";
  std::cout << "  --mode, -m <mode>      Mode: 'train', 'inference', or 'test' (overrides config file)\n";
  std::cout << "  --device, -d <device>  Device: 'cpu' or 'gpu' (overrides config file)\n";
  std::cout << "  --input, -i <file>     Path to JSON file with input values (inference mode, required)\n";
  std::cout << "  --samples, -s <file>   Path to JSON file with samples (train/test modes)\n";
  std::cout << "  --idx-data <file>      Path to IDX3 data file (alternative to --samples)\n";
  std::cout << "  --idx-labels <file>    Path to IDX1 labels file (requires --idx-data)\n";
  std::cout << "  --output, -o <file>    Output file (default: inference_<input>.json for inference mode)\n";
  std::cout << "  --verbose, -v          Print detailed initialization and processing info\n";
  std::cout << "  --help, -h             Show this help message\n";
}

int main(int argc, char *argv[]) {
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("ANN-CLI");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Artificial Neural Network CLI");
  parser.addHelpOption();

  // Config file option
  QCommandLineOption configOption(
    QStringList() << "c" << "config",
    "Path to JSON configuration file.",
    "file"
  );
  parser.addOption(configOption);

  // Mode option (train, inference, or test)
  QCommandLineOption modeOption(
    QStringList() << "m" << "mode",
    "Mode: 'train', 'inference', or 'test'.",
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

  // Input file for inference mode
  QCommandLineOption inputOption(
    QStringList() << "i" << "input",
    "Path to JSON file with input values for inference mode.",
    "file"
  );
  parser.addOption(inputOption);

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

  // Output file (train: model, inference: inference result with metadata)
  QCommandLineOption outputOption(
    QStringList() << "o" << "output",
    "Output file. Train mode: saves trained model. Inference mode: saves inference result with model metadata.",
    "file"
  );
  parser.addOption(outputOption);

  // Verbose option
  QCommandLineOption verboseOption(
    QStringList() << "v" << "verbose",
    "Print detailed initialization and processing information."
  );
  parser.addOption(verboseOption);

  parser.process(app);

  // Get verbose flag early - it controls all output
  bool verbose = parser.isSet(verboseOption);

  // Validate that --config is provided
  if (!parser.isSet(configOption)) {
    std::cerr << "Error: --config is required.\n\n";
    printUsage();
    return 1;
  }

  // Validate mode if provided
  if (parser.isSet(modeOption)) {
    QString modeStr = parser.value(modeOption).toLower();
    if (modeStr != "train" && modeStr != "inference" && modeStr != "test") {
      std::cerr << "Error: Mode must be 'train', 'inference', or 'test'.\n";
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

  try {
    ANN_CLI::Runner runner(parser, verbose);
    return runner.run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
