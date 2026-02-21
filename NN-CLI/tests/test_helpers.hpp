#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QString>
#include <QStringList>

#include <cmath>
#include <iostream>
#include <string>

extern int testsPassed;
extern int testsFailed;

#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

struct ProcessResult {
  int exitCode;
  QString stdOut;
  QString stdErr;
};

// Project root: parent of the build/ directory where binaries live
inline QString projectRoot() {
  return QCoreApplication::applicationDirPath() + "/..";
}

// Path to the NN-CLI binary (same build/ directory as test binary)
inline QString nncliPath() {
  return QCoreApplication::applicationDirPath() + "/NN-CLI";
}

// Path to a test fixture file
inline QString fixturePath(const QString& relativePath) {
  return projectRoot() + "/tests/fixtures/" + relativePath;
}

// Path to an example file
inline QString examplePath(const QString& relativePath) {
  return projectRoot() + "/examples/" + relativePath;
}

// Temp directory for test outputs
inline QString tempDir() {
  QString dir = QDir::temp().filePath("nncli_test");
  QDir().mkpath(dir);
  return dir;
}

// Clean up temp directory
inline void cleanupTemp() {
  QDir dir(QDir::temp().filePath("nncli_test"));
  if (dir.exists()) dir.removeRecursively();
}

// Run NN-CLI with arguments and capture output
inline ProcessResult runNNCLI(const QStringList& args, int timeoutMs = 120000) {
  QProcess process;
  process.setWorkingDirectory(projectRoot());
  process.start(nncliPath(), args);

  if (!process.waitForStarted(5000)) {
    return {-1, "", "Failed to start NN-CLI process"};
  }

  if (!process.waitForFinished(timeoutMs)) {
    process.kill();
    process.waitForFinished(3000);
    return {-2, "", "NN-CLI process timed out"};
  }

  return {
    process.exitCode(),
    QString::fromUtf8(process.readAllStandardOutput()),
    QString::fromUtf8(process.readAllStandardError())
  };
}

