#include "NN-Server_RequestHandler.hpp"
#include "NN-Server_HttpServer.hpp"
#include "NN-Server_ImageLoader.hpp"

#include <json.hpp>

#include <QElapsedTimer>

#include <QHostAddress>

#include <iostream>

namespace NN_Server
{

  // Convert IPv4-mapped IPv6 addresses (e.g. "::ffff:192.168.1.3") to plain IPv4.
  // Real IPv6 addresses are left unchanged.
  static std::string normalizeIp(const QHostAddress& address)
  {
    bool ok = false;
    quint32 ipv4 = address.toIPv4Address(&ok);

    if (ok) {
      return QHostAddress(ipv4).toString().toStdString();
    }

    return address.toString().toStdString();
  }

  RequestHandler::RequestHandler(qintptr socketDescriptor, std::shared_ptr<CorePool> pool, qint64 maxBodySize,
                                 std::shared_ptr<Logger> logger, HttpServer* server)
    : socketDescriptor(socketDescriptor),
      corePool(std::move(pool)),
      maxBodySize(maxBodySize),
      logger(std::move(logger)),
      server(server)
  {
  }

  //===================================================================================================================//

  void RequestHandler::run()
  {
    QTcpSocket socket;

    if (!socket.setSocketDescriptor(this->socketDescriptor)) {
      std::cerr << "Error: Failed to set socket descriptor.\n";

      if (this->server)
        this->server->requestFinished();

      return;
    }

    QElapsedTimer timer;
    timer.start();

    std::string clientIp = normalizeIp(socket.peerAddress());

    QByteArray requestData;
    bool rejected = false;
    int rejectedStatus = 0;

    while (socket.waitForReadyRead(5000)) {
      requestData.append(socket.readAll());

      if (requestData.contains("\r\n\r\n")) {
        int headerEnd = requestData.indexOf("\r\n\r\n");
        QByteArray headers = requestData.left(headerEnd);
        int contentLengthIdx = headers.toLower().indexOf("content-length:");

        if (contentLengthIdx >= 0) {
          int lineEnd = headers.indexOf("\r\n", contentLengthIdx);
          QByteArray clValue = headers.mid(contentLengthIdx + 15, lineEnd - contentLengthIdx - 15).trimmed();
          qint64 contentLength = clValue.toLongLong();

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

      if (this->maxBodySize > 0 && requestData.size() > this->maxBodySize + 8192) {
        sendJsonResponse(socket, 413,
                         R"({"error":"Request body too large. Maximum allowed: )" + std::to_string(this->maxBodySize) +
                           R"( bytes"})");
        rejected = true;
        rejectedStatus = 413;
        break;
      }
    }

    // If behind a reverse proxy (e.g. Caddy, Nginx, FRP), use the forwarded client IP
    if (requestData.contains("\r\n\r\n")) {
      QByteArray headers = requestData.left(requestData.indexOf("\r\n\r\n"));
      std::string forwarded = getHeader(headers, "x-forwarded-for");

      if (!forwarded.empty()) {
        // X-Forwarded-For may contain multiple IPs: "client, proxy1, proxy2"
        // The first one is the original client
        size_t comma = forwarded.find(',');

        if (comma != std::string::npos) {
          clientIp = forwarded.substr(0, comma);
        } else {
          clientIp = forwarded;
        }

        // Trim whitespace
        clientIp.erase(0, clientIp.find_first_not_of(' '));
        clientIp.erase(clientIp.find_last_not_of(' ') + 1);
      } else {
        std::string realIp = getHeader(headers, "x-real-ip");

        if (!realIp.empty()) {
          clientIp = realIp;
        }
      }
    }

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

    if (this->server)
      this->server->requestFinished();
  }

  //===================================================================================================================//

  int RequestHandler::processRequest(QTcpSocket& socket, const QByteArray& requestData)
  {
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

    if (method == "GET" && path == "/health") {
      sendJsonResponse(socket, 200, R"({"status":"ok"})");
      return 200;
    }

    if (method != "POST" || path != "/predict") {
      sendJsonResponse(socket, 404, R"({"error":"Not found. Use POST /predict"})");
      return 404;
    }

    int headerEnd = requestData.indexOf("\r\n\r\n");

    if (headerEnd < 0) {
      sendJsonResponse(socket, 400, R"({"error":"Missing request body"})");
      return 400;
    }

    QByteArray headers = requestData.left(headerEnd);
    QByteArray body = requestData.mid(headerEnd + 4);
    std::string contentType = getHeader(headers, "content-type");

    if (contentType.find("image/") != std::string::npos) {
      return processImagePredict(socket, body);
    } else {
      return processJsonPredict(socket, body);
    }
  }

  //===================================================================================================================//

  std::string RequestHandler::getHeader(const QByteArray& headers, const std::string& name)
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

  //===================================================================================================================//

  int RequestHandler::processJsonPredict(QTcpSocket& socket, const QByteArray& body)
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
      PredictionResult result = runPrediction(inputJson.at("input").get<std::vector<float>>());
      sendPredictResponse(socket, result);
      return 200;
    } catch (const std::exception& e) {
      std::string msg = "{\"error\":\"Prediction failed: " + std::string(e.what()) + "\"}";
      sendJsonResponse(socket, 500, msg);
      return 500;
    }
  }

  //===================================================================================================================//

  int RequestHandler::processImagePredict(QTcpSocket& socket, const QByteArray& imageData)
  {
    try {
      std::vector<float> inputVec = decodeImageInput(imageData);
      PredictionResult result = runPrediction(inputVec);
      sendPredictResponse(socket, result);
      return 200;
    } catch (const std::exception& e) {
      std::string msg = "{\"error\":\"Prediction failed: " + std::string(e.what()) + "\"}";
      sendJsonResponse(socket, 500, msg);
      return 500;
    }
  }

  //===================================================================================================================//

  RequestHandler::PredictionResult RequestHandler::runPrediction(const std::vector<float>& flatInput)
  {
    CoreHandle handle = this->corePool->acquire();

    try {
      PredictionResult result;

      if (this->corePool->networkType() == NetworkType::ANN) {
        Common::PredictResult<float> annResult = handle.annCore->predict(flatInput);
        result.output = std::move(annResult.output);
        result.logits = std::move(annResult.logits);
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
        Common::PredictResult<float> cnnResult = handle.cnnCore->predict(input);
        result.output = std::move(cnnResult.output);
        result.logits = std::move(cnnResult.logits);
      }

      this->corePool->release(handle);
      return result;
    } catch (...) {
      this->corePool->release(handle);
      throw;
    }
  }

  //===================================================================================================================//

  std::vector<float> RequestHandler::decodeImageInput(const QByteArray& imageData)
  {
    const InputConfig& inCfg = this->corePool->inputConfig();

    if (!inCfg.hasShape()) {
      throw std::runtime_error("Image input requires a model with a defined inputShape");
    }

    ulong targetC = inCfg.c;
    ulong targetH = inCfg.h;
    ulong targetW = inCfg.w;

    return ImageLoader::loadImageFromMemory(reinterpret_cast<const unsigned char*>(imageData.constData()),
                                            imageData.size(), static_cast<int>(targetC), static_cast<int>(targetH),
                                            static_cast<int>(targetW));
  }

  //===================================================================================================================//

  void RequestHandler::sendPredictResponse(QTcpSocket& socket, const PredictionResult& result)
  {
    const OutputConfig& outCfg = this->corePool->outputConfig();

    if (outCfg.isImage && outCfg.hasShape()) {
      // Image output (e.g. autoencoder): logits are not meaningful, return PNG only.
      std::vector<unsigned char> pngData = ImageLoader::saveImageToMemory(
        result.output, static_cast<int>(outCfg.c), static_cast<int>(outCfg.h), static_cast<int>(outCfg.w));
      sendBinaryResponse(socket, 200, "image/png", pngData);
    } else {
      nlohmann::json outputJson;
      outputJson["output"] = result.output;
      outputJson["logits"] = result.logits;
      sendJsonResponse(socket, 200, outputJson.dump());
    }
  }

  //===================================================================================================================//

  void RequestHandler::sendJsonResponse(QTcpSocket& socket, int statusCode, const std::string& body)
  {
    sendBinaryResponse(socket, statusCode, "application/json", std::vector<unsigned char>(body.begin(), body.end()));
  }

  //===================================================================================================================//

  void RequestHandler::sendBinaryResponse(QTcpSocket& socket, int statusCode, const std::string& contentType,
                                          const std::vector<unsigned char>& body)
  {
    std::string statusText;

    switch (statusCode) {
    case 200:
      statusText = "OK";
      break;
    case 400:
      statusText = "Bad Request";
      break;
    case 404:
      statusText = "Not Found";
      break;
    case 413:
      statusText = "Payload Too Large";
      break;
    case 503:
      statusText = "Service Unavailable";
      break;
    case 500:
      statusText = "Internal Server Error";
      break;
    default:
      statusText = "Unknown";
      break;
    }

    std::string header = "HTTP/1.1 " + std::to_string(statusCode) + " " + statusText +
                         "\r\n"
                         "Content-Type: " +
                         contentType +
                         "\r\n"
                         "Content-Length: " +
                         std::to_string(body.size()) +
                         "\r\n"
                         "Connection: close\r\n"
                         "\r\n";

    socket.write(header.c_str(), static_cast<qint64>(header.size()));
    socket.write(reinterpret_cast<const char*>(body.data()), static_cast<qint64>(body.size()));
  }

} // namespace NN_Server
