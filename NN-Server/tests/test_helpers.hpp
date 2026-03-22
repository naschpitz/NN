#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QProcess>
#include <QString>
#include <QTcpSocket>
#include <QThread>

#include <cmath>
#include <iostream>
#include <string>

extern int testsPassed;
extern int testsFailed;

// clang-format off
#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)
// clang-format on

// Project root: parent of the build/ directory where binaries live
inline QString projectRoot()
{
  return QCoreApplication::applicationDirPath() + "/..";
}

// Path to the NN-Server binary (same build/ directory as test binary)
inline QString serverBinPath()
{
  return QCoreApplication::applicationDirPath() + "/NN-Server";
}

// Path to a test fixture file
inline QString fixturePath(const QString& relativePath)
{
  return projectRoot() + "/tests/fixtures/" + relativePath;
}

// Path to a test image
inline QString imagePath(const QString& filename)
{
  return fixturePath("images/" + filename);
}

// Server configuration
constexpr int SERVER_PORT = 19876;
constexpr int POOL_SIZE   = 2;
constexpr int NUM_OUTPUT   = 11; // ISIC MILK10k has 11 output classes

/**
 * Simple HTTP response parsed from raw bytes.
 */
struct HttpResponse
{
    int statusCode = 0;
    QByteArray body;
    bool ok = false;
};

/**
 * Send a raw HTTP request to localhost:SERVER_PORT and return the response.
 * Blocks until the response is received or timeout expires.
 */
inline HttpResponse sendHttpRequest(const QByteArray& rawRequest, int timeoutMs = 120000)
{
  HttpResponse resp;
  QTcpSocket socket;
  socket.connectToHost("127.0.0.1", SERVER_PORT);

  if (!socket.waitForConnected(5000))
    return resp;

  socket.write(rawRequest);
  socket.flush();

  QByteArray data;
  while (socket.waitForReadyRead(timeoutMs)) {
    data.append(socket.readAll());
  }

  // Also grab anything remaining
  data.append(socket.readAll());
  socket.disconnectFromHost();

  if (data.isEmpty())
    return resp;

  // Parse status line
  int headerEnd = data.indexOf("\r\n\r\n");

  if (headerEnd < 0)
    return resp;

  QByteArray statusLine = data.left(data.indexOf("\r\n"));
  // "HTTP/1.1 200 OK"
  int firstSpace = statusLine.indexOf(' ');

  if (firstSpace < 0)
    return resp;

  int secondSpace = statusLine.indexOf(' ', firstSpace + 1);

  if (secondSpace < 0)
    secondSpace = statusLine.size();

  resp.statusCode = statusLine.mid(firstSpace + 1, secondSpace - firstSpace - 1).toInt();
  resp.body = data.mid(headerEnd + 4);
  resp.ok = true;
  return resp;
}

/**
 * Send a GET request.
 */
inline HttpResponse httpGet(const QString& path)
{
  QByteArray req = "GET " + path.toUtf8() + " HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Connection: close\r\n"
                   "\r\n";
  return sendHttpRequest(req);
}

/**
 * Send a POST request with a JSON body.
 */
inline HttpResponse httpPostJson(const QString& path, const QByteArray& jsonBody)
{
  QByteArray req = "POST " + path.toUtf8() + " HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Content-Type: application/json\r\n"
                   "Content-Length: " + QByteArray::number(jsonBody.size()) + "\r\n"
                   "Connection: close\r\n"
                   "\r\n" + jsonBody;
  return sendHttpRequest(req);
}

/**
 * Send a POST request with an image body.
 */
inline HttpResponse httpPostImage(const QString& path, const QByteArray& imageData)
{
  QByteArray req = "POST " + path.toUtf8() + " HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Content-Type: image/jpeg\r\n"
                   "Content-Length: " + QByteArray::number(imageData.size()) + "\r\n"
                   "Connection: close\r\n"
                   "\r\n" + imageData;
  return sendHttpRequest(req);
}

