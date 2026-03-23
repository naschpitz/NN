#pragma once

#include "NN-Server_CorePool.hpp"
#include "NN-Server_Logger.hpp"

#include <QRunnable>
#include <QTcpSocket>

#include <memory>
#include <string>
#include <vector>

namespace NN_Server
{

  class HttpServer;

  class RequestHandler : public QRunnable
  {
    public:
      RequestHandler(qintptr socketDescriptor, std::shared_ptr<CorePool> pool, qint64 maxBodySize,
                     std::shared_ptr<Logger> logger, HttpServer* server);

      void run() override;

    private:
      qintptr socketDescriptor;
      std::shared_ptr<CorePool> corePool;
      qint64 maxBodySize;
      std::shared_ptr<Logger> logger;
      HttpServer* server;

      int processRequest(QTcpSocket& socket, const QByteArray& requestData);

      static std::string getHeader(const QByteArray& headers, const std::string& name);

      int processJsonPredict(QTcpSocket& socket, const QByteArray& body);
      int processImagePredict(QTcpSocket& socket, const QByteArray& imageData);

      std::vector<float> runPrediction(const std::vector<float>& flatInput);
      std::vector<float> decodeImageInput(const QByteArray& imageData);

      void sendPredictResponse(QTcpSocket& socket, const std::vector<float>& output);
      static void sendJsonResponse(QTcpSocket& socket, int statusCode, const std::string& body);
      static void sendBinaryResponse(QTcpSocket& socket, int statusCode, const std::string& contentType,
                                     const std::vector<unsigned char>& body);
  };

} // namespace NN_Server
