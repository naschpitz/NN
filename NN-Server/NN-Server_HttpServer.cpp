#include "NN-Server_HttpServer.hpp"
#include "NN-Server_RequestHandler.hpp"

#include <QHostAddress>
#include <QTcpSocket>
#include <QThreadPool>

#include <iostream>

namespace NN_Server
{

  //===================================================================================================================//

  HttpServer::HttpServer(std::shared_ptr<CorePool> pool, qint64 maxBodySize, int maxQueueSize,
                         std::shared_ptr<Logger> logger, QObject* parent)
    : QTcpServer(parent),
      corePool(std::move(pool)),
      maxBodySize(maxBodySize),
      maxQueueSize(maxQueueSize),
      logger(std::move(logger))
  {
  }

  //===================================================================================================================//

  bool HttpServer::startListening(quint16 port)
  {
    if (!this->listen(QHostAddress::Any, port)) {
      std::cerr << "Error: Could not start server on port " << port << ": " << this->errorString().toStdString()
                << "\n";
      return false;
    }

    std::cout << "Listening on port " << port << "...\n\n";
    return true;
  }

  //===================================================================================================================//

  void HttpServer::requestFinished()
  {
    this->activeRequests.fetch_sub(1);
  }

  //===================================================================================================================//

  void HttpServer::incomingConnection(qintptr socketDescriptor)
  {
    if (this->maxQueueSize > 0 && this->activeRequests.load() >= this->maxQueueSize) {
      QTcpSocket socket;
      socket.setSocketDescriptor(socketDescriptor);

      std::string body = R"({"error":"Server busy. Too many queued requests"})";
      std::string response = "HTTP/1.1 503 Service Unavailable\r\n"
                             "Content-Type: application/json\r\n"
                             "Content-Length: " +
                             std::to_string(body.size()) +
                             "\r\n"
                             "Connection: close\r\n"
                             "\r\n" +
                             body;

      socket.write(response.c_str(), static_cast<qint64>(response.size()));
      socket.flush();
      socket.waitForBytesWritten(1000);
      socket.disconnectFromHost();

      if (this->logger) {
        // Normalize IPv4-mapped IPv6 addresses
        bool ok = false;
        quint32 ipv4 = socket.peerAddress().toIPv4Address(&ok);
        std::string ip = ok ? QHostAddress(ipv4).toString().toStdString()
                            : socket.peerAddress().toString().toStdString();
        this->logger->logRequest(ip, "?", "?", 503, 0.0);
      }

      return;
    }

    this->activeRequests.fetch_add(1);

    auto* handler = new RequestHandler(socketDescriptor, this->corePool, this->maxBodySize, this->logger, this);
    handler->setAutoDelete(true);
    QThreadPool::globalInstance()->start(handler);
  }

} // namespace NN_Server
