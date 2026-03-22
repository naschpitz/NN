#ifndef NN_SERVER_HTTPSERVER_HPP
#define NN_SERVER_HTTPSERVER_HPP

#include "NN-Server_CorePool.hpp"

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
      HttpServer(std::shared_ptr<CorePool> pool, QObject* parent = nullptr);

      bool startListening(quint16 port);

    protected:
      void incomingConnection(qintptr socketDescriptor) override;

    private:
      std::shared_ptr<CorePool> corePool;
  };

} // namespace NN_Server

#endif // NN_SERVER_HTTPSERVER_HPP

