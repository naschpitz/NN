#include "NN-Server_CorePool.hpp"
#include "NN-Server_HttpServer.hpp"
#include "NN-Server_Logger.hpp"

#include <json.hpp>

#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QCoreApplication>
#include <QFile>
#include <QThreadPool>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);

  QCommandLineParser parser;
  parser.setApplicationDescription("NN-Server — HTTP inference server");
  parser.addHelpOption();
  parser.addVersionOption();

  const QCommandLineOption modelPackageOption("model-package", "Path to trained model package (.nnmodel) to serve",
                                              "path");
  parser.addOption(modelPackageOption);

  const QCommandLineOption configOption("config", "Path to server settings config JSON file", "path");
  parser.addOption(configOption);

  parser.addPositionalArgument("config", "Path to server settings config JSON file (overrides --config)", "[config]");

  parser.process(app);

  // Determine config file path: positional arg > --config > default
  std::string configFilePath = "config.json";

  const QStringList posArgs = parser.positionalArguments();

  if (!posArgs.isEmpty()) {
    configFilePath = posArgs.first().toStdString();
  } else if (parser.isSet(configOption)) {
    configFilePath = parser.value(configOption).toStdString();
  }

  // Read and parse config.json
  QFile configFile(QString::fromStdString(configFilePath));

  if (!configFile.open(QIODevice::ReadOnly)) {
    std::cerr << "Error: Could not open configuration file: " << configFilePath << "\n";
    std::cerr << "       Pass the path as a positional argument or via --config.\n";
    std::cerr << "\n";
    std::cerr << "Usage: NN-Server --model-package <path> [config.json]\n";
    return 1;
  }

  nlohmann::json config;

  try {
    config = nlohmann::json::parse(configFile.readAll().toStdString());
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to parse " << configFilePath << ": " << e.what() << "\n";
    return 1;
  }

  configFile.close();

  // Model path: --model-package flag (preferred) → config["model_package"] fallback
  std::string modelPath;

  if (parser.isSet(modelPackageOption)) {
    modelPath = parser.value(modelPackageOption).toStdString();
  } else if (config.contains("model_package") && config["model_package"].is_string()) {
    modelPath = config["model_package"].get<std::string>();
  } else {
    std::cerr << "Error: No model path specified.\n";
    std::cerr << "       Provide --model-package <path> or set \"model_package\" in config.json.\n";
    parser.process(app);
    parser.showHelp(1);
  }

  // Server port (default 8080)
  quint16 port = 8080;

  if (config.contains("port")) {
    int portVal = config["port"].get<int>();

    if (portVal > 0 && portVal <= 65535) {
      port = static_cast<quint16>(portVal);
    } else {
      std::cerr << "Warning: Invalid port value " << portVal << ", using default 8080.\n";
    }
  }

  // Pool size (default = number of CPU cores)
  int poolSize = QThreadPool::globalInstance()->maxThreadCount();

  if (config.contains("poolSize")) {
    int val = config["poolSize"].get<int>();

    if (val > 0) {
      poolSize = val;
    } else {
      std::cerr << "Warning: Invalid poolSize value " << val << ", using default " << poolSize << ".\n";
    }
  }

  // Max request body size in megabytes (default: 10 MB)
  qint64 maxBodySizeMB = 10;

  if (config.contains("maxBodySize")) {
    qint64 val = config["maxBodySize"].get<qint64>();

    if (val > 0) {
      maxBodySizeMB = val;
    } else if (val == 0) {
      maxBodySizeMB = 0; // 0 = unlimited
    } else {
      std::cerr << "Warning: Invalid maxBodySize value " << val << ", using default 10 MB.\n";
    }
  }

  qint64 maxBodySize = maxBodySizeMB * 1024 * 1024; // Convert MB to bytes

  // Max queue size (default: 0 = unlimited)
  int maxQueueSize = 0;

  if (config.contains("maxQueueSize")) {
    int val = config["maxQueueSize"].get<int>();

    if (val >= 0) {
      maxQueueSize = val;
    } else {
      std::cerr << "Warning: Invalid maxQueueSize value " << val << ", using default (unlimited).\n";
    }
  }

  // Log file (optional)
  std::string logFile;

  if (config.contains("logFile") && config["logFile"].is_string()) {
    logFile = config["logFile"].get<std::string>();
  }

  // Max log size in gigabytes (default: 1 GB)
  qint64 maxLogSizeGB = 1;

  if (config.contains("maxLogSize")) {
    qint64 val = config["maxLogSize"].get<qint64>();

    if (val >= 0) {
      maxLogSizeGB = val;
    } else {
      std::cerr << "Warning: Invalid maxLogSize value " << val << ", using default 1 GB.\n";
    }
  }

  qint64 maxLogSizeBytes = maxLogSizeGB * 1024 * 1024 * 1024; // Convert GB to bytes

  std::cout << "NN-Server starting...\n";
  std::cout << "  Config:        " << configFilePath << "\n";
  std::cout << "  Model:         " << modelPath << "\n";
  std::cout << "  Port:         " << port << "\n";
  std::cout << "  Pool size:    " << poolSize << "\n";

  if (maxBodySizeMB > 0) {
    std::cout << "  Max body:     " << maxBodySizeMB << " MB\n";
  } else {
    std::cout << "  Max body:     unlimited\n";
  }

  if (maxQueueSize > 0) {
    std::cout << "  Max queue:    " << maxQueueSize << " requests\n";
  } else {
    std::cout << "  Max queue:    unlimited\n";
  }

  // Create logger (if logFile is set)
  std::shared_ptr<NN_Server::Logger> logger;

  if (!logFile.empty()) {
    logger = std::make_shared<NN_Server::Logger>(logFile, maxLogSizeBytes);
  }

  std::cout << "\n";

  // Create the core pool (loads models into memory, auto-detects output config)
  std::shared_ptr<NN_Server::CorePool> corePool;

  try {
    corePool = std::make_shared<NN_Server::CorePool>(modelPath, poolSize);
  } catch (const std::exception& e) {
    std::cerr << "Error loading model: " << e.what() << "\n";
    return 1;
  }

  const auto& inCfg = corePool->inputConfig();

  if (inCfg.isImage) {
    std::cout << "  Input:     image (" << inCfg.c << "x" << inCfg.h << "x" << inCfg.w << ")\n";
  } else {
    std::cout << "  Input:     vector (JSON)\n";
  }

  const auto& outCfg = corePool->outputConfig();

  if (outCfg.isImage) {
    std::cout << "  Output:    image (" << outCfg.c << "x" << outCfg.h << "x" << outCfg.w << ")\n";
  } else {
    std::cout << "  Output:    vector (JSON)\n";
  }

  std::cout << "\n";

  // Start the HTTP server
  NN_Server::HttpServer server(corePool, maxBodySize, maxQueueSize, logger);

  if (!server.startListening(port)) {
    return 1;
  }

  std::cout << "Ready to accept requests.\n";
  std::cout << "  POST /predict  — run prediction (JSON body or image upload)\n";
  std::cout << "  GET  /health   — health check\n";
  std::cout << "\n";

  return app.exec();
}
