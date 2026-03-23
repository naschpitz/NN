#ifndef NN_SERVER_HTTPSERVER_HPP
#define NN_SERVER_HTTPSERVER_HPP

#include "NN-Server_CorePool.hpp"
#include "NN-Server_Logger.hpp"

#include <QTcpServer>

#include <memory>

namespace NN_Server
{

  /**
   * Minimal HTTP server built on QTcpServer.
   * Accepts POST /predict with a JSON body containing an "input" array.
   * Dispatches prediction to a CorePool worker thread and returns the output as JSON.
   */
  class HttpServer : public QTcpServer
  {
      Q_OBJECT

    public:
      // maxBodySize in bytes; 0 = unlimited (default: 10 MB)
      HttpServer(std::shared_ptr<CorePool> pool, qint64 maxBodySize = 10 * 1024 * 1024,
                 std::shared_ptr<Logger> logger = nullptr, QObject* parent = nullptr);

      bool startListening(quint16 port);

    protected:
      void incomingConnection(qintptr socketDescriptor) override;

    private:
      std::shared_ptr<CorePool> corePool;
      qint64 maxBodySize;
      std::shared_ptr<Logger> logger;
  };

} // namespace NN_Server

#endif // NN_SERVER_HTTPSERVER_HPP
