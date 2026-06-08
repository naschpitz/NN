#include "NN-Server_Logger.hpp"

#include <QMutexLocker>

#include <iostream>

namespace NN_Server
{

  Logger::Logger(const std::string& filePath, qint64 maxSizeBytes) : maxSizeBytes(maxSizeBytes)
  {
    if (filePath.empty())
      return;

    this->file.setFileName(QString::fromStdString(filePath));

    if (!this->file.open(QIODevice::ReadWrite | QIODevice::Append)) {
      std::cerr << "Warning: Could not open log file: " << filePath << ". Logging disabled.\n";
      return;
    }

    this->currentPos = this->file.size();

    std::cout << "  Log file:     " << filePath << "\n";

    if (maxSizeBytes > 0) {
      std::cout << "  Max log size: " << (maxSizeBytes / (1024 * 1024 * 1024)) << " GB\n";
    } else {
      std::cout << "  Max log size: unlimited\n";
    }
  }

  Logger::~Logger()
  {
    if (this->file.isOpen())
      this->file.close();
  }

  void Logger::logRequest(const std::string& clientIp, const std::string& method, const std::string& path,
                          int statusCode, double durationMs)
  {
    if (!this->file.isOpen())
      return;

    // Format: [2026-03-23 14:30:05.123 -03:00] 192.168.1.10 POST /predict 200 45.2ms
    std::string timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz t").toStdString();

    // Format duration with '.' decimal separator regardless of locale
    int whole = static_cast<int>(durationMs);
    int frac = static_cast<int>((durationMs - whole) * 10 + 0.5) % 10;
    std::string durationStr = std::to_string(whole) + "." + std::to_string(frac) + "ms";

    std::string line = "[" + timestamp + "] " + clientIp + " " + method + " " + path + " " +
                       std::to_string(statusCode) + " " + durationStr + "\n";

    this->writeLine(line);
  }

  void Logger::writeLine(const std::string& line)
  {
    QMutexLocker locker(&this->mutex);

    qint64 lineSize = static_cast<qint64>(line.size());

    // Check if we need to wrap around
    if (this->maxSizeBytes > 0 && (this->currentPos + lineSize) > this->maxSizeBytes) {
      this->file.seek(0);
      this->currentPos = 0;
      this->wrapped = true;
    }

    this->file.write(line.c_str(), lineSize);
    this->file.flush();
    this->currentPos += lineSize;

    // If we've wrapped, pad the rest of the current line area to avoid
    // leftover text from the previous cycle
    if (this->wrapped) {
      qint64 nextNewline = this->currentPos;

      // Read ahead to find the next newline from previous content
      // and overwrite it with spaces to keep the log clean
      QByteArray peek = this->file.peek(512);
      int nlPos = peek.indexOf('\n');

      if (nlPos >= 0) {
        QByteArray padding(nlPos + 1, ' ');
        padding[nlPos] = '\n';
        this->file.write(padding);
        this->file.seek(this->currentPos); // seek back to where we were
      }
    }
  }

} // namespace NN_Server
