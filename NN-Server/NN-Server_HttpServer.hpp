#ifndef NN_SERVER_HTTPSERVER_HPP
#define NN_SERVER_HTTPSERVER_HPP

#include "NN-Server_CorePool.hpp"
#include "NN-Server_Logger.hpp"

#include <QTcpServer>

#include <atomic>
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
      // maxQueueSize: max concurrent requests (0 = unlimited)
      HttpServer(std::shared_ptr<CorePool> pool, qint64 maxBodySize = 10 * 1024 * 1024, int maxQueueSize = 0,
                 std::shared_ptr<Logger> logger = nullptr, QObject* parent = nullptr);

      bool startListening(quint16 port);

      // Called by RequestHandler when a request finishes
      void requestFinished();

    protected:
      void incomingConnection(qintptr socketDescriptor) override;

    private:
      std::shared_ptr<CorePool> corePool;
      qint64 maxBodySize;
      int maxQueueSize;
      std::shared_ptr<Logger> logger;
      std::atomic<int> activeRequests{0};
  };

} // namespace NN_Server

#endif // NN_SERVER_HTTPSERVER_HPP
