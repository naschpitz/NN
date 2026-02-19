#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDebug>
#include <QDir>
#include <QFileInfo>

#include <ANN_Core.hpp>
#include <ANN_CoreMode.hpp>
#include <ANN_CoreType.hpp>
#include <ANN_Utils.hpp>

#include "ANN-CLI_Loader.hpp"
#include "ANN-CLI_ProgressBar.hpp"
#include "ANN-CLI_Utils.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>

// Generate default output filename with timestamp
std::string generateTimestampedFilename() {
  auto now = std::chrono::system_clock::now();
  std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
  std::tm* localTime = std::localtime(&nowTime);

  std::ostringstream oss;
  oss << "trained_model_"
      << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S")
      << ".json";

  return oss.str();
}

// Generate default output path based on config file location
std::string generateDefaultOutputPath(const QString& configPath) {
  QFileInfo configInfo(configPath);
  QDir configDir = configInfo.absoluteDir();
  QDir outputDir(configDir.filePath("output"));

  // Create output directory if it doesn't exist
  if (!outputDir.exists()) {
    configDir.mkdir("output");
  }

  QString outputPath = outputDir.filePath(QString::fromStdString(generateTimestampedFilename()));
  return outputPath.toStdString();
}

void printUsage() {
  std::cout << "ANN-CLI - Artificial Neural Network Command Line Interface\n\n";
  std::cout << "Usage:\n";
  std::cout << "  ANN-CLI --config <file> [options]           # Train mode (mode from config or --mode)\n";
  std::cout << "  ANN-CLI --model <file> --input <file>       # Run mode (inference)\n";
  std::cout << "  ANN-CLI --model <file> --mode test [options] # Test mode (evaluation)\n\n";
  std::cout << "Options:\n";
  std::cout << "  --mode, -m <mode>      Mode: 'train', 'run', or 'test' (overrides config file)\n";
  std::cout << "  --config, -c <file>    Path to JSON configuration file\n";
  std::cout << "  --model <file>         Path to trained model file\n";
  std::cout << "  --device, -d <device>  Device: 'cpu' or 'gpu' (overrides config file)\n";
  std::cout << "  --input, -i <file>     Path to JSON file with input values (run mode)\n";
  std::cout << "  --samples, -s <file>   Path to JSON file with samples (train/test modes)\n";
  std::cout << "  --idx-data <file>      Path to IDX3 data file (alternative to --samples)\n";
  std::cout << "  --idx-labels <file>    Path to IDX1 labels file (requires --idx-data)\n";
  std::cout << "  --output, -o <file>    Output file for saving trained model (train mode)\n";
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

  // Mode option (train, run, or test)
  QCommandLineOption modeOption(
    QStringList() << "m" << "mode",
    "Mode: 'train', 'run', or 'test'.",
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

  // Input file for run mode
  QCommandLineOption inputOption(
    QStringList() << "i" << "input",
    "Path to JSON file with input values for run mode.",
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

  // Output file for saving trained model
  QCommandLineOption outputOption(
    QStringList() << "o" << "output",
    "Output file for saving trained model (default: <config_dir>/output/trained_model_[timestamp].json).",
    "file"
  );
  parser.addOption(outputOption);

  // Model file for run/test modes
  QCommandLineOption modelOption(
    QStringList() << "model",
    "Path to trained model file (for run/test modes).",
    "file"
  );
  parser.addOption(modelOption);

  parser.process(app);

  // Validate that at least one of --config or --model is provided
  if (!parser.isSet(configOption) && !parser.isSet(modelOption)) {
    std::cerr << "Error: Either --config or --model must be provided.\n\n";
    printUsage();
    return 1;
  }

  // Build optional mode override (only if --mode was explicitly provided)
  std::optional<ANN::CoreModeType> modeOverride;
  QString modeStr;  // For display and validation

  if (parser.isSet(modeOption)) {
    modeStr = parser.value(modeOption).toLower();
    if (modeStr != "train" && modeStr != "run" && modeStr != "test") {
      std::cerr << "Error: Mode must be 'train', 'run', or 'test'.\n";
      return 1;
    }
    if (modeStr == "train") {
      modeOverride = ANN::CoreModeType::TRAIN;
    } else if (modeStr == "test") {
      modeOverride = ANN::CoreModeType::TEST;
    } else {
      modeOverride = ANN::CoreModeType::RUN;
    }
  }

  // Build optional device override (only if --device was explicitly provided)
  std::optional<ANN::DeviceType> deviceOverride;

  if (parser.isSet(deviceOption)) {
    QString deviceStr = parser.value(deviceOption).toLower();
    if (deviceStr != "cpu" && deviceStr != "gpu") {
      std::cerr << "Error: Device must be 'cpu' or 'gpu'.\n";
      return 1;
    }
    deviceOverride = (deviceStr == "gpu") ? ANN::DeviceType::GPU : ANN::DeviceType::CPU;
  }

  // Determine which file to use based on what's provided
  // --config is for training (new models), --model is for run/test (pre-trained models)
  bool useConfigFile = parser.isSet(configOption);
  bool useModelFile = parser.isSet(modelOption);

  if (useConfigFile && useModelFile) {
    std::cerr << "Error: Cannot use both --config and --model. Choose one.\n";
    return 1;
  }

  try {
    std::unique_ptr<ANN::Core<float>> core;
    QString configPath;  // Used for train mode (needed later for default output path)
    ANN::CoreConfig<float> coreConfig;

    // Get device string for display (from CLI if provided, otherwise will be from config file)
    std::string deviceDisplay = deviceOverride.has_value()
        ? (deviceOverride.value() == ANN::DeviceType::GPU ? "gpu (CLI)" : "cpu (CLI)")
        : "from config file";

    // Get mode string for display (from CLI if provided, otherwise will be from config file)
    std::string modeDisplay;
    if (modeOverride.has_value()) {
      switch (modeOverride.value()) {
        case ANN::CoreModeType::TRAIN: modeDisplay = "train (CLI)"; break;
        case ANN::CoreModeType::TEST:  modeDisplay = "test (CLI)";  break;
        case ANN::CoreModeType::RUN:   modeDisplay = "run (CLI)";   break;
        default:                       modeDisplay = "unknown (CLI)"; break;
      }
    } else {
      modeDisplay = "from config file";
    }

    if (useConfigFile) {
      // Load config file (architecture + training settings, optionally pre-trained weights)
      configPath = parser.value(configOption);
      std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
      std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";

      coreConfig = ANN_CLI::Loader::loadConfig(configPath.toStdString(), modeOverride, deviceOverride);
    } else {
      // Load model file (architecture + trained weights)
      QString modelPath = parser.value(modelOption);
      std::cout << "Loading model from: " << modelPath.toStdString() << "\n";
      std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";

      coreConfig = ANN_CLI::Loader::loadModel(modelPath.toStdString(), deviceOverride);
    }

    // Get the actual mode from the loaded config (may have come from file or CLI override)
    bool isTrainMode = (coreConfig.coreModeType == ANN::CoreModeType::TRAIN);
    bool isTestMode = (coreConfig.coreModeType == ANN::CoreModeType::TEST);

    core = ANN::Core<float>::makeCore(coreConfig);

    if (isTrainMode) {
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

      // Set up progress callback with ProgressBar instance
      ANN_CLI::ProgressBar progressBar;

      core->setTrainingCallback([&progressBar](const ANN::TrainingProgress<float>& progress) {
        progressBar.update(progress);
      });

      core->train(samples);
      std::cout << "\nTraining completed.\n";

      // Save the trained model (uses default timestamped filename in output/ folder if not specified)
      std::string outputPathStr;

      if (parser.isSet(outputOption)) {
        outputPathStr = parser.value(outputOption).toStdString();
      } else {
        outputPathStr = generateDefaultOutputPath(configPath);
      }

      ANN::Utils<float>::save(*core, outputPathStr);
      std::cout << "Model saved to: " << outputPathStr << "\n";

    } else if (isTestMode) {
      // Test mode - evaluate model on test samples
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
        std::cout << "Loading test samples from JSON: " << samplesPath.toStdString() << "\n";
        samples = ANN_CLI::Loader::loadSamples(samplesPath.toStdString());
      } else if (hasIdxData) {
        // Load from IDX format
        if (!hasIdxLabels) {
          std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
          return 1;
        }

        QString idxDataPath = parser.value(idxDataOption);
        QString idxLabelsPath = parser.value(idxLabelsOption);

        std::cout << "Loading test samples from IDX:\n";
        std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
        std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";

        samples = ANN_CLI::Utils<float>::loadIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
      } else {
        std::cerr << "Error: Test mode requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
        return 1;
      }

      std::cout << "Loaded " << samples.size() << " test samples.\n";
      std::cout << "Running evaluation...\n";

      ANN::TestResult<float> result = core->test(samples);

      std::cout << "\nTest Results:\n";
      std::cout << "  Samples evaluated: " << result.numSamples << "\n";
      std::cout << "  Total loss:        " << result.totalLoss << "\n";
      std::cout << "  Average loss:      " << result.averageLoss << "\n";

    } else {
      // Run mode (default for non-train, non-test)
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
