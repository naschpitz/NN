#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <ANN_Core.hpp>
#include <ANN_Mode.hpp>
#include <ANN_Device.hpp>
#include <ANN_Utils.hpp>

#include "ANN-CLI_Loader.hpp"
#include "ANN-CLI_ProgressBar.hpp"
#include "ANN-CLI_Utils.hpp"

#include <json.hpp>

#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>

// Generate default output filename with training info
std::string generateTrainingFilename(ulong epochs, ulong samples, float loss) {
  std::ostringstream oss;
  oss << "trained_model_"
      << epochs << "_"
      << samples << "_"
      << std::fixed << std::setprecision(6) << loss
      << ".json";

  return oss.str();
}

// Generate default output path based on input file location (samples or idx-data)
std::string generateDefaultOutputPath(const QString& inputFilePath, ulong epochs, ulong samples, float loss) {
  QFileInfo inputInfo(inputFilePath);
  QDir inputDir = inputInfo.absoluteDir();
  QDir outputDir(inputDir.filePath("output"));

  // Create output directory if it doesn't exist
  if (!outputDir.exists()) {
    inputDir.mkdir("output");
  }

  QString outputPath = outputDir.filePath(QString::fromStdString(generateTrainingFilename(epochs, samples, loss)));
  return outputPath.toStdString();
}

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

  // Build optional mode override (only if --mode was explicitly provided)
  std::optional<ANN::ModeType> modeOverride;
  QString modeStr;  // For display and validation

  if (parser.isSet(modeOption)) {
    modeStr = parser.value(modeOption).toLower();
    if (modeStr != "train" && modeStr != "inference" && modeStr != "test") {
      std::cerr << "Error: Mode must be 'train', 'inference', or 'test'.\n";
      return 1;
    }
    if (modeStr == "train") {
      modeOverride = ANN::ModeType::TRAIN;
    } else if (modeStr == "test") {
      modeOverride = ANN::ModeType::TEST;
    } else {
      modeOverride = ANN::ModeType::INFERENCE;
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

  try {
    std::unique_ptr<ANN::Core<float>> core;
    QString configPath = parser.value(configOption);
    ANN::CoreConfig<float> coreConfig;

    // Get device string for display (from CLI if provided, otherwise will be from config file)
    std::string deviceDisplay = deviceOverride.has_value()
        ? (deviceOverride.value() == ANN::DeviceType::GPU ? "gpu (CLI)" : "cpu (CLI)")
        : "from config file";

    // Get mode string for display (from CLI if provided, otherwise will be from config file)
    std::string modeDisplay;
    if (modeOverride.has_value()) {
      switch (modeOverride.value()) {
        case ANN::ModeType::TRAIN:     modeDisplay = "train (CLI)";     break;
        case ANN::ModeType::TEST:      modeDisplay = "test (CLI)";      break;
        case ANN::ModeType::INFERENCE: modeDisplay = "inference (CLI)"; break;
        default:                       modeDisplay = "unknown (CLI)";   break;
      }
    } else {
      modeDisplay = "from config file";
    }

    if (verbose) {
      std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
      std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
    }

    coreConfig = ANN_CLI::Loader::loadConfig(configPath.toStdString(), modeOverride, deviceOverride);
    coreConfig.verbose = verbose;

    // Get the actual mode from the loaded config (may have come from file or CLI override)
    bool isTrainMode = (coreConfig.modeType == ANN::ModeType::TRAIN);
    bool isTestMode = (coreConfig.modeType == ANN::ModeType::TEST);

    core = ANN::Core<float>::makeCore(coreConfig);

    if (isTrainMode) {
      // Training mode
      ANN::Samples<float> samples;
      QString inputFilePath;  // Track input file path for default output location

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
        inputFilePath = samplesPath;
        if (verbose) std::cout << "Loading training samples from JSON: " << samplesPath.toStdString() << "\n";
        samples = ANN_CLI::Loader::loadSamples(samplesPath.toStdString());
      } else if (hasIdxData) {
        // Load from IDX format
        if (!hasIdxLabels) {
          std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
          return 1;
        }

        QString idxDataPath = parser.value(idxDataOption);
        QString idxLabelsPath = parser.value(idxLabelsOption);
        inputFilePath = idxDataPath;

        if (verbose) {
          std::cout << "Loading training samples from IDX:\n";
          std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
          std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
        }

        samples = ANN_CLI::Utils<float>::loadIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
      } else {
        std::cerr << "Error: Training requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
        return 1;
      }

      if (verbose) std::cout << "Loaded " << samples.size() << " training samples.\n";

      if (verbose) std::cout << "Starting training...\n";

      // Set up progress callback with ProgressBar instance
      ANN_CLI::ProgressBar progressBar;

      core->setTrainingCallback([&progressBar](const ANN::TrainingProgress<float>& progress) {
        progressBar.update(progress);
      });

      core->train(samples);
      std::cout << "\nTraining completed.\n";

      // Get training info for filename generation
      const auto& trainingConfig = core->getTrainingConfig();
      const auto& trainingMetadata = core->getTrainingMetadata();

      // Save the trained model (uses training info in filename if not specified)
      std::string outputPathStr;

      if (parser.isSet(outputOption)) {
        outputPathStr = parser.value(outputOption).toStdString();
      } else {
        outputPathStr = generateDefaultOutputPath(
          inputFilePath,
          trainingConfig.numEpochs,
          trainingMetadata.numSamples,
          trainingMetadata.finalLoss
        );
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
        if (verbose) std::cout << "Loading test samples from JSON: " << samplesPath.toStdString() << "\n";
        samples = ANN_CLI::Loader::loadSamples(samplesPath.toStdString());
      } else if (hasIdxData) {
        // Load from IDX format
        if (!hasIdxLabels) {
          std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
          return 1;
        }

        QString idxDataPath = parser.value(idxDataOption);
        QString idxLabelsPath = parser.value(idxLabelsOption);

        if (verbose) {
          std::cout << "Loading test samples from IDX:\n";
          std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
          std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
        }

        samples = ANN_CLI::Utils<float>::loadIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
      } else {
        std::cerr << "Error: Test mode requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
        return 1;
      }

      if (verbose) {
        std::cout << "Loaded " << samples.size() << " test samples.\n";
        std::cout << "Running evaluation...\n";
      }

      ANN::TestResult<float> result = core->test(samples);

      // Test results are always shown (not verbose)
      std::cout << "\nTest Results:\n";
      std::cout << "  Samples evaluated: " << result.numSamples << "\n";
      std::cout << "  Total loss:        " << result.totalLoss << "\n";
      std::cout << "  Average loss:      " << result.averageLoss << "\n";

    } else {
      // Inference mode (default for non-train, non-test)
      if (!parser.isSet(inputOption)) {
        std::cerr << "Error: --input option is required for inference mode.\n";
        return 1;
      }

      QString inputPath = parser.value(inputOption);

      // Generate default output path if not specified: <input_dir>/output/inference_[input_basename].json
      QString outputPath;
      if (parser.isSet(outputOption)) {
        outputPath = parser.value(outputOption);
      } else {
        QFileInfo inputInfo(inputPath);
        QString baseName = inputInfo.completeBaseName();  // filename without extension
        QDir inputDir = inputInfo.absoluteDir();
        QDir outputDir(inputDir.filePath("output"));

        // Create output directory if it doesn't exist
        if (!outputDir.exists()) {
          inputDir.mkdir("output");
        }

        outputPath = outputDir.filePath("inference_" + baseName + ".json");
      }

      if (verbose) std::cout << "Loading input from: " << inputPath.toStdString() << "\n";

      ANN::Input<float> input = ANN_CLI::Loader::loadInput(inputPath.toStdString());

      if (verbose) {
        std::cout << "Running ANN with input: ";
        for (size_t i = 0; i < input.size(); ++i) {
          std::cout << input[i];
          if (i < input.size() - 1) std::cout << ", ";
        }
        std::cout << "\n";
      }

      ANN::Output<float> output = core->run(input);

      // Get run metadata from core (timing captured inside Core::run())
      const auto& runMetadata = core->getRunMetadata();

      nlohmann::ordered_json resultJson;

      // Add run metadata first
      nlohmann::ordered_json runMetadataJson;
      runMetadataJson["startTime"] = runMetadata.startTime;
      runMetadataJson["endTime"] = runMetadata.endTime;
      runMetadataJson["durationSeconds"] = runMetadata.durationSeconds;
      runMetadataJson["durationFormatted"] = runMetadata.durationFormatted;
      resultJson["runMetadata"] = runMetadataJson;

      // Add inference result
      resultJson["output"] = output;

      // Write to file
      QFile outputFile(outputPath);
      if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        std::cerr << "Error: Failed to open output file: " << outputPath.toStdString() << "\n";
        return 1;
      }

      std::string jsonStr = resultJson.dump(2);
      outputFile.write(jsonStr.c_str(), jsonStr.size());
      outputFile.close();

      std::cout << "Inference result saved to: " << outputPath.toStdString() << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
