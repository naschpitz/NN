#include "test_helpers.hpp"
#include "NN-Server_Logger.hpp"

#include <QDir>
#include <QFile>
#include <QTextStream>

static QString tempLogDir()
{
  QString dir = QDir::temp().filePath("nnserver_logger_test");
  QDir().mkpath(dir);
  return dir;
}

static QString readFile(const QString& path)
{
  QFile f(path);

  if (!f.open(QIODevice::ReadOnly))
    return "";

  return f.readAll();
}

// ---------------------------------------------------------------------------

static void testLogCreatesFile()
{
  std::cout << "  testLogCreatesFile... " << std::flush;

  QString logPath = tempLogDir() + "/test_create.log";
  QFile::remove(logPath);

  {
    NN_Server::Logger logger(logPath.toStdString(), 0);
    CHECK(logger.isEnabled(), "log_create: logger is enabled");
    logger.logRequest("10.0.0.1", "GET", "/health", 200, 1.5);
  }

  CHECK(QFile::exists(logPath), "log_create: file exists");

  QString content = readFile(logPath);
  CHECK(content.contains("10.0.0.1"), "log_create: contains client IP");
  CHECK(content.contains("GET"), "log_create: contains method");
  CHECK(content.contains("/health"), "log_create: contains path");
  CHECK(content.contains("200"), "log_create: contains status code");
  CHECK(content.contains("1.5ms"), "log_create: contains duration");

  std::cout << std::endl;
}

static void testLogFormat()
{
  std::cout << "  testLogFormat... " << std::flush;

  QString logPath = tempLogDir() + "/test_format.log";
  QFile::remove(logPath);

  {
    NN_Server::Logger logger(logPath.toStdString(), 0);
    logger.logRequest("192.168.1.50", "POST", "/predict", 200, 42.3);
  }

  QString content = readFile(logPath);

  // Should match: [YYYY-MM-DD HH:MM:SS.mmm TZ] IP METHOD PATH STATUS DURATIONms
  CHECK(content.contains("["), "log_format: has opening bracket");
  CHECK(content.contains("]"), "log_format: has closing bracket");
  CHECK(content.contains("192.168.1.50"), "log_format: has IP");
  CHECK(content.contains("POST /predict 200 42.3ms"), "log_format: has method path status duration");

  // Verify timezone is present (e.g. UTC, -03:00, +05:30, etc.)
  // The timestamp format is: [YYYY-MM-DD HH:MM:SS.mmm TZ]
  int closeBracket = content.indexOf(']');
  QString timestamp = content.mid(1, closeBracket - 1); // strip [ and ]
  // Timestamp should have more than just "YYYY-MM-DD HH:MM:SS.mmm" (23 chars)
  CHECK(timestamp.length() > 23, "log_format: timestamp includes timezone (length=" +
        std::to_string(timestamp.length()) + ")");

  std::cout << std::endl;
}

static void testLogAppendsOnRestart()
{
  std::cout << "  testLogAppendsOnRestart... " << std::flush;

  QString logPath = tempLogDir() + "/test_append.log";
  QFile::remove(logPath);

  // First session
  {
    NN_Server::Logger logger(logPath.toStdString(), 0);
    logger.logRequest("10.0.0.1", "GET", "/health", 200, 1.0);
  }

  // Second session — should append, not overwrite
  {
    NN_Server::Logger logger(logPath.toStdString(), 0);
    logger.logRequest("10.0.0.2", "POST", "/predict", 200, 5.0);
  }

  QString content = readFile(logPath);
  CHECK(content.contains("10.0.0.1"), "log_append: first session entry present");
  CHECK(content.contains("10.0.0.2"), "log_append: second session entry present");

  // Count lines
  int lineCount = content.count('\n');
  CHECK(lineCount == 2, "log_append: exactly 2 lines (got " + std::to_string(lineCount) + ")");

  std::cout << std::endl;
}

static void testLogMultipleEntries()
{
  std::cout << "  testLogMultipleEntries... " << std::flush;

  QString logPath = tempLogDir() + "/test_multi.log";
  QFile::remove(logPath);

  {
    NN_Server::Logger logger(logPath.toStdString(), 0);
    logger.logRequest("10.0.0.1", "GET", "/health", 200, 0.1);
    logger.logRequest("10.0.0.2", "POST", "/predict", 200, 50.0);
    logger.logRequest("10.0.0.3", "GET", "/missing", 404, 0.2);
    logger.logRequest("10.0.0.4", "POST", "/predict", 500, 10.0);
  }

  QString content = readFile(logPath);
  int lineCount = content.count('\n');
  CHECK(lineCount == 4, "log_multi: 4 lines (got " + std::to_string(lineCount) + ")");
  CHECK(content.contains("10.0.0.1") && content.contains("10.0.0.4"), "log_multi: first and last entries present");
  CHECK(content.contains("404"), "log_multi: contains 404 status");
  CHECK(content.contains("500"), "log_multi: contains 500 status");

  std::cout << std::endl;
}

static void testLogCircularWrap()
{
  std::cout << "  testLogCircularWrap... " << std::flush;

  QString logPath = tempLogDir() + "/test_circular.log";
  QFile::remove(logPath);

  // Use a small max size to force wrapping
  // Each log line is roughly ~80 bytes
  qint64 maxSize = 200; // room for ~2 lines

  {
    NN_Server::Logger logger(logPath.toStdString(), maxSize);
    logger.logRequest("10.0.0.1", "GET", "/first", 200, 1.0);
    logger.logRequest("10.0.0.2", "GET", "/second", 200, 2.0);
    logger.logRequest("10.0.0.3", "GET", "/third", 200, 3.0); // should wrap
  }

  QFile f(logPath);
  f.open(QIODevice::ReadOnly);
  qint64 fileSize = f.size();
  f.close();

  // File should not exceed maxSize significantly
  CHECK(fileSize <= maxSize + 200, "log_circular: file size bounded (got " + std::to_string(fileSize) + ")");

  // The latest entry should be present
  QString content = readFile(logPath);
  CHECK(content.contains("/third"), "log_circular: latest entry present after wrap");

  std::cout << std::endl;
}

static void testLogCircularAppendsAfterRestart()
{
  std::cout << "  testLogCircularAppendsAfterRestart... " << std::flush;

  QString logPath = tempLogDir() + "/test_circular_restart.log";
  QFile::remove(logPath);

  qint64 maxSize = 500;

  // First session
  {
    NN_Server::Logger logger(logPath.toStdString(), maxSize);
    logger.logRequest("10.0.0.1", "GET", "/session1", 200, 1.0);
  }

  QFile f1(logPath);
  f1.open(QIODevice::ReadOnly);
  qint64 sizeAfterFirst = f1.size();
  f1.close();

  // Second session — should append from where the file ended
  {
    NN_Server::Logger logger(logPath.toStdString(), maxSize);
    logger.logRequest("10.0.0.2", "GET", "/session2", 200, 2.0);
  }

  QFile f2(logPath);
  f2.open(QIODevice::ReadOnly);
  qint64 sizeAfterSecond = f2.size();
  f2.close();

  CHECK(sizeAfterSecond > sizeAfterFirst, "log_circular_restart: file grew after second session");

  QString content = readFile(logPath);
  CHECK(content.contains("/session1"), "log_circular_restart: first session entry present");
  CHECK(content.contains("/session2"), "log_circular_restart: second session entry present");

  std::cout << std::endl;
}

static void testLogDisabledWithEmptyPath()
{
  std::cout << "  testLogDisabledWithEmptyPath... " << std::flush;

  NN_Server::Logger logger("", 0);
  CHECK(!logger.isEnabled(), "log_disabled: logger not enabled with empty path");

  // Should not crash
  logger.logRequest("10.0.0.1", "GET", "/health", 200, 1.0);

  std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void runLoggerTests()
{
  testLogCreatesFile();
  testLogFormat();
  testLogAppendsOnRestart();
  testLogMultipleEntries();
  testLogCircularWrap();
  testLogCircularAppendsAfterRestart();
  testLogDisabledWithEmptyPath();
}

