#include "NN-Server_HttpServer.hpp"
#include "NN-Server_ImageLoader.hpp"

#include <QElapsedTimer>
#include <QTcpSocket>
#include <QThreadPool>
#include <QRunnable>

#include <json.hpp>

#include <iostream>

namespace NN_Server
{

  //===================================================================================================================//
  // RequestHandler — QRunnable that processes one HTTP request in a thread pool thread
  //===================================================================================================================//

  class RequestHandler : public QRunnable
  {
    public:
      RequestHandler(qintptr socketDescriptor, std::shared_ptr<CorePool> pool, qint64 maxBodySize,
                     std::shared_ptr<Logger> logger)
        : socketDescriptor(socketDescriptor), corePool(std::move(pool)), maxBodySize(maxBodySize),
          logger(std::move(logger))
      {
      }

      void run() override
      {
        QTcpSocket socket;

        if (!socket.setSocketDescriptor(this->socketDescriptor)) {
          std::cerr << "Error: Failed to set socket descriptor.\n";
          return;
        }

        QElapsedTimer timer;
        timer.start();

        std::string clientIp = socket.peerAddress().toString().toStdString();

        // Read the full HTTP request (wait up to 5 seconds)
        QByteArray requestData;
        bool rejected = false;
        int rejectedStatus = 0;

        while (socket.waitForReadyRead(5000)) {
          requestData.append(socket.readAll());

          // Check if we have the full request (look for end of body)
          if (requestData.contains("\r\n\r\n")) {
            // Check for Content-Length
            int headerEnd = requestData.indexOf("\r\n\r\n");
            QByteArray headers = requestData.left(headerEnd);
            int contentLengthIdx = headers.toLower().indexOf("content-length:");

            if (contentLengthIdx >= 0) {
              int lineEnd = headers.indexOf("\r\n", contentLengthIdx);
              QByteArray clValue = headers.mid(contentLengthIdx + 15, lineEnd - contentLengthIdx - 15).trimmed();
              qint64 contentLength = clValue.toLongLong();

              // Reject early if Content-Length exceeds the limit
              if (this->maxBodySize > 0 && contentLength > this->maxBodySize) {
                sendJsonResponse(socket, 413,
                  R"({"error":"Request body too large. Maximum allowed: )" +
                  std::to_string(this->maxBodySize) + R"( bytes"})");
                rejected = true;
                rejectedStatus = 413;
                break;
              }

              int bodyStart = headerEnd + 4;
              qint64 bodyLength = requestData.size() - bodyStart;

              if (bodyLength >= contentLength)
                break;
            } else {
              break;
            }
          }

          // Also reject if accumulated data exceeds the limit (no Content-Length or chunked)
          if (this->maxBodySize > 0 && requestData.size() > this->maxBodySize + 8192) { // 8KB header allowance
            sendJsonResponse(socket, 413,
              R"({"error":"Request body too large. Maximum allowed: )" +
              std::to_string(this->maxBodySize) + R"( bytes"})");
            rejected = true;
            rejectedStatus = 413;
            break;
          }
        }

        // Parse method and path for logging
        std::string method = "?";
        std::string path = "?";
        int statusCode = 0;

        if (!requestData.isEmpty()) {
          int firstSpace = requestData.indexOf(' ');

          if (firstSpace > 0) {
            method = requestData.left(firstSpace).toStdString();
            int secondSpace = requestData.indexOf(' ', firstSpace + 1);

            if (secondSpace > 0)
              path = requestData.mid(firstSpace + 1, secondSpace - firstSpace - 1).toStdString();
          }
        }

        if (rejected) {
          statusCode = rejectedStatus;
        } else {
          statusCode = this->processRequest(socket, requestData);
        }

        double durationMs = timer.nsecsElapsed() / 1e6;

        if (this->logger)
          this->logger->logRequest(clientIp, method, path, statusCode, durationMs);

        socket.flush();
        socket.waitForBytesWritten(3000);
        socket.disconnectFromHost();

        if (socket.state() != QAbstractSocket::UnconnectedState) {
          socket.waitForDisconnected(3000);
        }
      }

    private:
      qintptr socketDescriptor;
      std::shared_ptr<CorePool> corePool;
      qint64 maxBodySize;
      std::shared_ptr<Logger> logger;

      // Returns the HTTP status code for logging
      int processRequest(QTcpSocket& socket, const QByteArray& requestData)
      {
        // Parse HTTP request line
        int firstLineEnd = requestData.indexOf("\r\n");

        if (firstLineEnd < 0) {
          sendJsonResponse(socket, 400, R"({"error":"Malformed HTTP request"})");
          return 400;
        }

        QByteArray requestLine = requestData.left(firstLineEnd);
        int firstSpace = requestLine.indexOf(' ');
        int secondSpace = requestLine.indexOf(' ', firstSpace + 1);

        if (firstSpace < 0 || secondSpace < 0) {
          sendJsonResponse(socket, 400, R"({"error":"Malformed HTTP request"})");
          return 400;
        }

        QByteArray method = requestLine.left(firstSpace);
        QByteArray path = requestLine.mid(firstSpace + 1, secondSpace - firstSpace - 1);

        // Health check endpoint
        if (method == "GET" && path == "/health") {
          sendJsonResponse(socket, 200, R"({"status":"ok"})");
          return 200;
        }

        // Only accept POST /predict
        if (method != "POST" || path != "/predict") {
          sendJsonResponse(socket, 404, R"({"error":"Not found. Use POST /predict"})");
          return 404;
        }

        // Split headers and body
        int headerEnd = requestData.indexOf("\r\n\r\n");

        if (headerEnd < 0) {
          sendJsonResponse(socket, 400, R"({"error":"Missing request body"})");
          return 400;
        }

        QByteArray headers = requestData.left(headerEnd);
        QByteArray body = requestData.mid(headerEnd + 4);

        // Determine content type
        std::string contentType = getHeader(headers, "content-type");

        if (contentType.find("image/") != std::string::npos) {
          return processImagePredict(socket, body);
        } else {
          // Default: JSON
          return processJsonPredict(socket, body);
        }
      }

      //-- Header parsing --//

      static std::string getHeader(const QByteArray& headers, const std::string& name)
      {
        QByteArray lowerHeaders = headers.toLower();
        QByteArray lowerName = QByteArray::fromStdString(name);
        int idx = lowerHeaders.indexOf(lowerName + ":");

        if (idx < 0)
          return "";

        int valueStart = idx + lowerName.size() + 1;
        int lineEnd = headers.indexOf("\r\n", valueStart);

        if (lineEnd < 0)
          lineEnd = headers.size();

        return headers.mid(valueStart, lineEnd - valueStart).trimmed().toStdString();
      }

      //-- JSON predict --//

      int processJsonPredict(QTcpSocket& socket, const QByteArray& body)
      {
        nlohmann::json inputJson;

        try {
          inputJson = nlohmann::json::parse(body.toStdString());
        } catch (const std::exception& e) {
          std::string msg = "{\"error\":\"Invalid JSON: " + std::string(e.what()) + "\"}";
          sendJsonResponse(socket, 400, msg);
          return 400;
        }

        if (!inputJson.contains("input") || !inputJson.at("input").is_array()) {
          sendJsonResponse(socket, 400, R"delim({"error":"Request must contain an 'input' array"})delim");
          return 400;
        }

        try {
          std::vector<float> output = runPrediction(inputJson.at("input").get<std::vector<float>>());
          sendPredictResponse(socket, output);
          return 200;
        } catch (const std::exception& e) {
          std::string msg = "{\"error\":\"Prediction failed: " + std::string(e.what()) + "\"}";
          sendJsonResponse(socket, 500, msg);
          return 500;
        }
      }

      //-- Image predict (raw image body) --//

      int processImagePredict(QTcpSocket& socket, const QByteArray& imageData)
      {
        try {
          std::vector<float> inputVec = decodeImageInput(imageData);
          std::vector<float> output = runPrediction(inputVec);
          sendPredictResponse(socket, output);
          return 200;
        } catch (const std::exception& e) {
          std::string msg = "{\"error\":\"Prediction failed: " + std::string(e.what()) + "\"}";
          sendJsonResponse(socket, 500, msg);
          return 500;
        }
      }

      //-- Core prediction logic --//

      std::vector<float> runPrediction(const std::vector<float>& flatInput)
      {
        CoreHandle handle = this->corePool->acquire();

        try {
          std::vector<float> output;

          if (this->corePool->networkType() == NetworkType::ANN) {
            output = handle.annCore->predict(flatInput);
          } else {
            const CNN::Shape3D& shape = handle.cnnCore->getInputShape();

            if (flatInput.size() != shape.size()) {
              std::string msg = "Input size (" + std::to_string(flatInput.size()) +
                                ") does not match expected shape size (" + std::to_string(shape.size()) + ")";
              this->corePool->release(handle);
              throw std::runtime_error(msg);
            }

            CNN::Input<float> input(shape);
            input.data = flatInput;
            output = handle.cnnCore->predict(input);
          }

          this->corePool->release(handle);
          return output;
        } catch (...) {
          this->corePool->release(handle);
          throw;
        }
      }

      //-- Image decoding --//

      std::vector<float> decodeImageInput(const QByteArray& imageData)
      {
        const InputConfig& inCfg = this->corePool->inputConfig();

        if (!inCfg.hasShape()) {
          throw std::runtime_error("Image input requires a model with a defined inputShape");
        }

        ulong targetC = inCfg.c;
        ulong targetH = inCfg.h;
        ulong targetW = inCfg.w;

        return ImageLoader::loadImageFromMemory(
          reinterpret_cast<const unsigned char*>(imageData.constData()),
          imageData.size(),
          static_cast<int>(targetC), static_cast<int>(targetH), static_cast<int>(targetW));
      }

      //-- Response helpers --//

      void sendPredictResponse(QTcpSocket& socket, const std::vector<float>& output)
      {
        const OutputConfig& outCfg = this->corePool->outputConfig();

        if (outCfg.isImage && outCfg.hasShape()) {
          // Return output as a PNG image
          std::vector<unsigned char> pngData = ImageLoader::saveImageToMemory(
            output, static_cast<int>(outCfg.c), static_cast<int>(outCfg.h), static_cast<int>(outCfg.w));
          sendBinaryResponse(socket, 200, "image/png", pngData);
        } else {
          // Return output as JSON
          nlohmann::json outputJson;
          outputJson["output"] = output;
          sendJsonResponse(socket, 200, outputJson.dump());
        }
      }

      static void sendJsonResponse(QTcpSocket& socket, int statusCode, const std::string& body)
      {
        sendBinaryResponse(socket, statusCode, "application/json",
                           std::vector<unsigned char>(body.begin(), body.end()));
      }

      static void sendBinaryResponse(QTcpSocket& socket, int statusCode, const std::string& contentType,
                                     const std::vector<unsigned char>& body)
      {
        std::string statusText;

        switch (statusCode) {
        case 200: statusText = "OK"; break;
        case 400: statusText = "Bad Request"; break;
        case 404: statusText = "Not Found"; break;
        case 500: statusText = "Internal Server Error"; break;
        default: statusText = "Unknown"; break;
        }

        std::string header = "HTTP/1.1 " + std::to_string(statusCode) + " " + statusText + "\r\n"
                             "Content-Type: " + contentType + "\r\n"
                             "Content-Length: " + std::to_string(body.size()) + "\r\n"
                             "Connection: close\r\n"
                             "\r\n";

        socket.write(header.c_str(), static_cast<qint64>(header.size()));
        socket.write(reinterpret_cast<const char*>(body.data()), static_cast<qint64>(body.size()));
      }

  };

  //===================================================================================================================//
  // HttpServer
  //===================================================================================================================//

  HttpServer::HttpServer(std::shared_ptr<CorePool> pool, qint64 maxBodySize,
                         std::shared_ptr<Logger> logger, QObject* parent)
    : QTcpServer(parent), corePool(std::move(pool)), maxBodySize(maxBodySize),
      logger(std::move(logger))
  {
  }

  //===================================================================================================================//

  bool HttpServer::startListening(quint16 port)
  {
    if (!this->listen(QHostAddress::Any, port)) {
      std::cerr << "Error: Failed to start server on port " << port << ": "
                << this->errorString().toStdString() << "\n";
      return false;
    }

    std::cout << "Server listening on port " << port << "\n";
    return true;
  }

  //===================================================================================================================//

  void HttpServer::incomingConnection(qintptr socketDescriptor)
  {
    auto* handler = new RequestHandler(socketDescriptor, this->corePool, this->maxBodySize, this->logger);
    handler->setAutoDelete(true);
    QThreadPool::globalInstance()->start(handler);
  }

  //===================================================================================================================//

} // namespace NN_Server

