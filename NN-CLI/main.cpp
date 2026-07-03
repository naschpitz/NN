#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>

#include "NN-CLI_App.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelPackage.hpp"

#include <iostream>
#include <string>

void printUsage()
{
  std::cout << "NN-CLI - Neural Network Command Line Interface (ANN + CNN)\n\n";
  std::cout << "Usage:\n";
  std::cout << "  NN-CLI [options]\n\n";
  std::cout << "Examples:\n";
  std::cout << "  train:    NN-CLI --mode train --model <config.json> [options]\n";
  std::cout << "  predict:  NN-CLI --mode predict --model-package <model.nnmodel> [options]\n";
  std::cout << "  test:     NN-CLI --mode test --model-package <model.nnmodel> [options]\n";
  std::cout << "  calibrate: NN-CLI --mode calibrate --model-package <model.nnmodel> [options]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --model <path>           Model architecture JSON (for train)\n";
  std::cout << "  --model-package <path>   Trained model package (for predict, test, calibrate)\n";
  std::cout << "  --mode, -m <mode>        Mode: 'train', 'predict', 'test', or 'calibrate'\n";
  std::cout << "  --device, -d <device>    Device: 'cpu' or 'gpu' (overrides config file)\n";
  std::cout << "  --input, -i <file>       Path to JSON file with batch inputs (predict mode, required)\n";
  std::cout << "  --input-type <type>      Input data type: 'vector' or 'image' (overrides config file)\n";
  std::cout << "  --samples, -s <file>     Path to JSON file with samples (train/test modes)\n";
  std::cout << "  --idx-data <file>        Path to IDX3 data file (alternative to --samples)\n";
  std::cout << "  --idx-labels <file>      Path to IDX1 labels file (requires --idx-data)\n";
  std::cout << "  --output, -o <dir>       Output directory (default: <input>/output). predict_*.json, test_*.json, "
               "threshold.json\n";
  std::cout << "  --output-type <type>     Output data type: 'vector' or 'image' (overrides config file)\n";
  std::cout << "  --log-level, -l <lvl>    Log level: quiet, error, warning, info, debug (default: error)\n";
  std::cout << "  --gpu-profile            Enable OpenCL GPU kernel profiling (adds ~12% overhead)\n";
  std::cout
    << "  --gpu-profile-dump <dir> With --gpu-profile: dump per-kernel timings to <dir> (requires --gpu-profile)\n";
  std::cout << "\nCalibrate-mode options:\n";
  std::cout << "  --id-images <dir>        Directory of in-distribution images (recursed) [required]\n";
  std::cout << "  --ood-dir <dir>          OOD images directory (default: <cwd>/extern-datasets/ood)\n";
  std::cout << "  --id-sample-count <N>    Random subsample size for ID set (default 500)\n";
  std::cout << "  --ood-sample-count <N>   Random subsample size for OOD set (default 1500)\n";
  std::cout << "  --id-percentile <P>      ID percentile used as the threshold (default 95)\n";
  std::cout << "  --no-fetch               Don't auto-download OOD if --ood-dir is empty (default: fetch)\n";
  std::cout << "  --help, -h               Show this help message\n";
}

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);
  QCoreApplication::setApplicationName("NN-CLI");
  QCoreApplication::setApplicationVersion("1.0");

  QCommandLineParser parser;
  parser.setApplicationDescription("Neural Network CLI (ANN + CNN)");
  parser.addHelpOption();

  // Model file options
  QCommandLineOption modelOption("model", "Path to model architecture JSON file (for training)", "path");
  parser.addOption(modelOption);

  QCommandLineOption modelPackageOption(
    "model-package", "Path to trained model package (.nnmodel) (for predict, test, calibrate)", "path");
  parser.addOption(modelPackageOption);

  // Mode option (train, predict, test, or calibrate)
  QCommandLineOption modeOption(QStringList() << "m" << "mode", "Mode: 'train', 'predict', 'test', or 'calibrate'.",
                                "mode");
  parser.addOption(modeOption);

  // Calibrate-mode options
  QCommandLineOption idImagesOption("id-images", "Calibrate: directory of in-distribution images (recursed).", "dir");
  parser.addOption(idImagesOption);

  QCommandLineOption oodDirOption(
    "ood-dir", "Calibrate: OOD root (default: <cwd>/extern-datasets/ood). Auto-fetched if empty.", "dir");
  parser.addOption(oodDirOption);

  QCommandLineOption idSampleCountOption("id-sample-count", "Calibrate: ID subsample size (default 500).", "N");
  parser.addOption(idSampleCountOption);

  QCommandLineOption oodSampleCountOption("ood-sample-count", "Calibrate: OOD subsample size (default 1500).", "N");
  parser.addOption(oodSampleCountOption);

  QCommandLineOption idPercentileOption("id-percentile", "Calibrate: ID percentile used as the threshold (default 95).",
                                        "P");
  parser.addOption(idPercentileOption);

  QCommandLineOption noFetchOption("no-fetch", "Calibrate: don't auto-download OOD even if --ood-dir is empty.");
  parser.addOption(noFetchOption);

  // Device option (cpu or gpu)
  QCommandLineOption deviceOption(QStringList() << "d" << "device", "Device: 'cpu' or 'gpu' (default: cpu).", "device",
                                  "cpu");
  parser.addOption(deviceOption);

  // Input file for predict mode
  QCommandLineOption inputOption(QStringList() << "i" << "input",
                                 "Path to JSON file with input values for predict mode.", "file");
  parser.addOption(inputOption);

  // Input type option (vector or image)
  QCommandLineOption inputTypeOption(QStringList() << "input-type",
                                     "Input data type: 'vector' or 'image' (overrides config file).", "type");
  parser.addOption(inputTypeOption);

  // Samples file for training/testing (JSON format)
  QCommandLineOption samplesOption(QStringList() << "s" << "samples",
                                   "Path to JSON file with samples (for train/test modes).", "file");
  parser.addOption(samplesOption);

  // IDX data file for training (IDX3 format)
  QCommandLineOption idxDataOption(QStringList() << "idx-data", "Path to IDX3 data file (alternative to --samples).",
                                   "file");
  parser.addOption(idxDataOption);

  // IDX labels file for training (IDX1 format)
  QCommandLineOption idxLabelsOption(QStringList() << "idx-labels", "Path to IDX1 labels file (requires --idx-data).",
                                     "file");
  parser.addOption(idxLabelsOption);

  // Output file (train: model, predict: predict result with metadata)
  QCommandLineOption outputOption(
    QStringList() << "o" << "output",
    "Output file. Train mode: saves trained model. Predict mode: saves predict result with model metadata.", "file");
  parser.addOption(outputOption);

  // Output type option (vector or image)
  QCommandLineOption outputTypeOption(QStringList() << "output-type",
                                      "Output data type: 'vector' or 'image' (overrides config file).", "type");
  parser.addOption(outputTypeOption);

  // Log level option
  QCommandLineOption logLevelOption(QStringList() << "l" << "log-level",
                                    "Log level: quiet, error, warning, info, debug (default: error).", "level",
                                    "error");
  parser.addOption(logLevelOption);

  QCommandLineOption gpuProfileOption("gpu-profile", "Enable OpenCL GPU kernel profiling (adds ~12% overhead).");
  parser.addOption(gpuProfileOption);

  QCommandLineOption gpuProfileDumpOption("gpu-profile-dump", "With --gpu-profile: dump per-kernel timings to <dir>.",
                                          "dir");
  parser.addOption(gpuProfileDumpOption);

  parser.process(app);

  // Validate that at least one model option is provided
  if (!parser.isSet(modelOption) && !parser.isSet(modelPackageOption)) {
    std::cerr << "Error: --model or --model-package is required.\n\n";
    printUsage();
    return 1;
  }

  // Validate that --model and --model-package are not both set
  if (parser.isSet(modelOption) && parser.isSet(modelPackageOption)) {
    std::cerr << "Error: --model and --model-package are mutually exclusive.\n";
    return 1;
  }

  // Validate mode-specific requirements
  if (parser.isSet(modeOption)) {
    QString modeStr = parser.value(modeOption).toLower();

    if (modeStr == "train" && !parser.isSet(modelOption)) {
      std::cerr << "Error: --model is required for train mode.\n";
      return 1;
    }

    if (modeStr == "train" && parser.isSet(modelPackageOption)) {
      std::cerr << "Error: --model-package cannot be used with --mode train.\n";
      return 1;
    }

    if ((modeStr == "predict" || modeStr == "test" || modeStr == "calibrate") && !parser.isSet(modelPackageOption)) {
      std::cerr << "Error: --model-package is required for " << modeStr.toStdString() << " mode.\n";
      return 1;
    }

    if ((modeStr == "predict" || modeStr == "test" || modeStr == "calibrate") && parser.isSet(modelOption)) {
      std::cerr << "Error: --model cannot be used with --mode " << modeStr.toStdString() << ".\n";
      return 1;
    }
  }

  // Validate mode if provided
  if (parser.isSet(modeOption)) {
    QString modeStr = parser.value(modeOption).toLower();

    if (modeStr != "train" && modeStr != "predict" && modeStr != "test" && modeStr != "calibrate") {
      std::cerr << "Error: Mode must be 'train', 'predict', 'test', or 'calibrate'.\n";
      return 1;
    }

    // Non-train modes require a .nnmodel package (not a plain .json config)
    if ((modeStr == "predict" || modeStr == "test" || modeStr == "calibrate") && parser.isSet(modelPackageOption)) {
      QString configFilePath = parser.value(modelPackageOption);

      if (!NN_CLI::ModelPackage::isPackage(configFilePath.toStdString())) {
        std::cerr << "Error: " << modeStr.toStdString() << " mode requires a .nnmodel package (not a plain .json).\n";
        std::cerr << "The plain JSON config format is no longer supported for this mode.\n";
        return 1;
      }
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

  // Validate gpu-profile-dump requires gpu-profile
  if (parser.isSet("gpu-profile-dump") && !parser.isSet("gpu-profile")) {
    std::cerr << "Error: --gpu-profile-dump requires --gpu-profile to be set.\n";
    return 1;
  }

  // Parse log level
  NN_CLI::LogLevel logLevel = NN_CLI::LogLevel::ERROR;

  if (parser.isSet(logLevelOption)) {
    QString levelStr = parser.value(logLevelOption).toLower();

    if (levelStr == "quiet")
      logLevel = NN_CLI::LogLevel::QUIET;
    else if (levelStr == "error")
      logLevel = NN_CLI::LogLevel::ERROR;
    else if (levelStr == "warning")
      logLevel = NN_CLI::LogLevel::WARNING;
    else if (levelStr == "info")
      logLevel = NN_CLI::LogLevel::INFO;
    else if (levelStr == "debug")
      logLevel = NN_CLI::LogLevel::DEBUG;
    else {
      std::cerr << "Error: Log level must be 'quiet', 'error', 'warning', 'info', or 'debug'.\n";
      return 1;
    }
  }

  try {
    NN_CLI::App app(parser, logLevel);
    return app.run();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
