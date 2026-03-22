#include "NN-Server_CorePool.hpp"
#include "NN-Server_HttpServer.hpp"

#include <json.hpp>

#include <QCoreApplication>
#include <QFile>
#include <QThreadPool>

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);

  // Determine config file path: first CLI argument or default "config.json"
  std::string configFilePath = "config.json";

  if (argc > 1) {
    configFilePath = argv[1];
  }

  // Read and parse config.json
  QFile configFile(QString::fromStdString(configFilePath));

  if (!configFile.open(QIODevice::ReadOnly)) {
    std::cerr << "Error: Could not open configuration file: " << configFilePath << "\n";
    std::cerr << "       Pass the path as an argument or place a config.json in the current directory.\n";
    std::cerr << "\n";
    std::cerr << "Usage: NN-Server [config.json]\n";
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

  // Model path (required)
  if (!config.contains("model") || !config["model"].is_string()) {
    std::cerr << "Error: config.json must contain a \"model\" field with the path to the model file.\n";
    return 1;
  }

  std::string modelPath = config["model"].get<std::string>();

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
      std::cerr << "Warning: Invalid poolSize value " << val
                << ", using default " << poolSize << ".\n";
    }
  }

  // Max request body size (default: 10 MB)
  qint64 maxBodySize = 10 * 1024 * 1024;

  if (config.contains("maxBodySize")) {
    qint64 val = config["maxBodySize"].get<qint64>();

    if (val > 0) {
      maxBodySize = val;
    } else if (val == 0) {
      maxBodySize = 0; // 0 = unlimited
    } else {
      std::cerr << "Warning: Invalid maxBodySize value " << val
                << ", using default 10 MB.\n";
    }
  }

  std::cout << "NN-Server starting...\n";
  std::cout << "  Config:       " << configFilePath << "\n";
  std::cout << "  Model:        " << modelPath << "\n";
  std::cout << "  Port:         " << port << "\n";
  std::cout << "  Pool size:    " << poolSize << "\n";

  if (maxBodySize > 0) {
    std::cout << "  Max body:     " << maxBodySize << " bytes (" << (maxBodySize / (1024 * 1024)) << " MB)\n";
  } else {
    std::cout << "  Max body:     unlimited\n";
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
  NN_Server::HttpServer server(corePool, maxBodySize);

  if (!server.startListening(port)) {
    return 1;
  }

  std::cout << "Ready to accept requests.\n";
  std::cout << "  POST /predict  — run prediction (JSON body or image upload)\n";
  std::cout << "  GET  /health   — health check\n";
  std::cout << "\n";

  return app.exec();
}

