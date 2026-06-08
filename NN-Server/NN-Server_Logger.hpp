#pragma once

#include <QDateTime>
#include <QFile>
#include <QMutex>
#include <QString>

#include <memory>
#include <string>

namespace NN_Server
{

  /**
   * Thread-safe file logger with circular writing.
   *
   * When the log file reaches maxSizeBytes, it seeks back to the beginning
   * and overwrites from the start, creating a circular log.
   */
  class Logger
  {
    public:
      // maxSizeBytes: maximum log file size in bytes. 0 = unlimited.
      Logger(const std::string& filePath, qint64 maxSizeBytes);
      ~Logger();

      // Log a request: date, time, client IP, method, path, HTTP status, duration
      void logRequest(const std::string& clientIp, const std::string& method, const std::string& path, int statusCode,
                      double durationMs);

      // Check if logging is enabled
      bool isEnabled() const
      {
        return this->file.isOpen();
      }

    private:
      void writeLine(const std::string& line);

      QFile file;
      QMutex mutex;
      qint64 maxSizeBytes;
      qint64 currentPos = 0;
      bool wrapped = false; // true if we've wrapped around at least once
  };

} // namespace NN_Server
