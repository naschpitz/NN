#include "NN-Server_CorePool.hpp"
#include "NN-Server_HttpServer.hpp"

#include <QCoreApplication>
#include <QThreadPool>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char* argv[])
{
  QCoreApplication app(argc, argv);

  // Read configuration from environment variables
  const char* configPathEnv = std::getenv("NN_MODEL_CONFIG");

  if (configPathEnv == nullptr || std::string(configPathEnv).empty()) {
    std::cerr << "Error: NN_MODEL_CONFIG environment variable is not set.\n";
    std::cerr << "       Set it to the path of a JSON model configuration file.\n";
    return 1;
  }

  std::string configPath(configPathEnv);

  // Server port (default 8080)
  quint16 port = 8080;
  const char* portEnv = std::getenv("NN_SERVER_PORT");

  if (portEnv != nullptr) {
    int portVal = std::atoi(portEnv);

    if (portVal > 0 && portVal <= 65535) {
      port = static_cast<quint16>(portVal);
    } else {
      std::cerr << "Warning: Invalid NN_SERVER_PORT value '" << portEnv << "', using default 8080.\n";
    }
  }

  // Pool size (default = number of CPU cores)
  int poolSize = QThreadPool::globalInstance()->maxThreadCount();
  const char* poolSizeEnv = std::getenv("NN_SERVER_POOL_SIZE");

  if (poolSizeEnv != nullptr) {
    int val = std::atoi(poolSizeEnv);

    if (val > 0) {
      poolSize = val;
    } else {
      std::cerr << "Warning: Invalid NN_SERVER_POOL_SIZE value '" << poolSizeEnv
                << "', using default " << poolSize << ".\n";
    }
  }

  // Output shape (optional — when set, output is returned as an image)
  // Format: "CxHxW" e.g. "3x256x256"
  NN_Server::OutputConfig outputConfig;
  const char* outputShapeEnv = std::getenv("NN_OUTPUT_SHAPE");

  if (outputShapeEnv != nullptr) {
    std::string shapeStr(outputShapeEnv);

    // Parse CxHxW
    size_t firstX = shapeStr.find('x');
    size_t secondX = shapeStr.find('x', firstX + 1);

    if (firstX != std::string::npos && secondX != std::string::npos) {
      outputConfig.c = std::stoul(shapeStr.substr(0, firstX));
      outputConfig.h = std::stoul(shapeStr.substr(firstX + 1, secondX - firstX - 1));
      outputConfig.w = std::stoul(shapeStr.substr(secondX + 1));
      outputConfig.isImage = true;
    } else {
      std::cerr << "Warning: Invalid NN_OUTPUT_SHAPE format '" << shapeStr
                << "'. Expected CxHxW (e.g. 3x256x256).\n";
    }
  }

  std::cout << "NN-Server starting...\n";
  std::cout << "  Config:    " << configPath << "\n";
  std::cout << "  Port:      " << port << "\n";
  std::cout << "  Pool size: " << poolSize << "\n";

  if (outputConfig.isImage) {
    std::cout << "  Output:    image (" << outputConfig.c << "x" << outputConfig.h << "x" << outputConfig.w << ")\n";
  } else {
    std::cout << "  Output:    vector (JSON)\n";
  }

  std::cout << "\n";

  // Create the core pool (loads models into memory)
  std::shared_ptr<NN_Server::CorePool> corePool;

  try {
    corePool = std::make_shared<NN_Server::CorePool>(configPath, poolSize);
    corePool->setOutputConfig(outputConfig);
  } catch (const std::exception& e) {
    std::cerr << "Error loading model: " << e.what() << "\n";
    return 1;
  }

  std::cout << "\n";

  // Start the HTTP server
  NN_Server::HttpServer server(corePool);

  if (!server.startListening(port)) {
    return 1;
  }

  std::cout << "Ready to accept requests.\n";
  std::cout << "  POST /predict  — run prediction (JSON body or image upload)\n";
  std::cout << "  GET  /health   — health check\n";
  std::cout << "\n";

  return app.exec();
}

