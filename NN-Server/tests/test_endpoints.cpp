#include "test_helpers.hpp"

#include <json.hpp>

#include <QDir>
#include <QFile>
#include <QProcess>
#include <QTemporaryDir>
#include <QThread>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Server lifecycle helpers
// ---------------------------------------------------------------------------

static QProcess* serverProcess = nullptr;
static QTemporaryDir* tmpDir = nullptr;

static bool startServer()
{
  // Create a temporary config.json file for the server
  tmpDir = new QTemporaryDir();

  if (!tmpDir->isValid()) {
    std::cerr << "Failed to create temporary directory" << std::endl;
    return false;
  }

  QString configPath = tmpDir->path() + "/config.json";
  nlohmann::json config;
  config["model"] = fixturePath("checkpoint_E-150_L-0.029486.json").toStdString();
  config["port"] = SERVER_PORT;
  config["poolSize"] = POOL_SIZE;
  config["maxBodySize"] = MAX_BODY_SIZE_MB;

  QFile configFile(configPath);

  if (!configFile.open(QIODevice::WriteOnly)) {
    std::cerr << "Failed to write test config.json" << std::endl;
    return false;
  }

  configFile.write(QByteArray::fromStdString(config.dump(2)));
  configFile.close();

  serverProcess = new QProcess();
  serverProcess->setWorkingDirectory(projectRoot());
  serverProcess->start(serverBinPath(), QStringList() << configPath);

  if (!serverProcess->waitForStarted(5000)) {
    std::cerr << "Failed to start server process" << std::endl;
    delete serverProcess;
    serverProcess = nullptr;
    return false;
  }

  // Poll /health until the server is ready (model loading can be slow)
  for (int i = 0; i < 120; i++) {
    QThread::sleep(1);
    HttpResponse resp = httpGet("/health");

    if (resp.ok && resp.statusCode == 200) {
      std::cout << "Server is ready." << std::endl;
      return true;
    }

    // Check if process died
    if (serverProcess->state() == QProcess::NotRunning) {
      std::cerr << "Server exited early. stdout:\n"
                << serverProcess->readAllStandardOutput().toStdString() << "\nstderr:\n"
                << serverProcess->readAllStandardError().toStdString() << std::endl;
      delete serverProcess;
      serverProcess = nullptr;
      return false;
    }
  }

  std::cerr << "Server did not become ready in time" << std::endl;
  serverProcess->kill();
  serverProcess->waitForFinished(5000);
  delete serverProcess;
  serverProcess = nullptr;
  return false;
}

static void stopServer()
{
  if (serverProcess) {
    serverProcess->terminate();

    if (!serverProcess->waitForFinished(10000)) {
      serverProcess->kill();
      serverProcess->waitForFinished(5000);
    }

    delete serverProcess;
    serverProcess = nullptr;
  }

  if (tmpDir) {
    delete tmpDir;
    tmpDir = nullptr;
  }
}

// ---------------------------------------------------------------------------
// Helper: read an image file into a QByteArray
// ---------------------------------------------------------------------------

static QByteArray readImageFile(const QString& path)
{
  QFile f(path);

  if (!f.open(QIODevice::ReadOnly))
    return {};

  return f.readAll();
}

// ---------------------------------------------------------------------------
// Individual tests
// ---------------------------------------------------------------------------

static void testHealth()
{
  std::cout << "  testHealth... " << std::flush;

  HttpResponse resp = httpGet("/health");
  CHECK(resp.ok, "health: got response");
  CHECK(resp.statusCode == 200, "health: status 200");

  nlohmann::json body = nlohmann::json::parse(resp.body.toStdString());
  CHECK(body.contains("status") && body["status"] == "ok", "health: status=ok");

  std::cout << std::endl;
}

static void testNotFound()
{
  std::cout << "  testNotFound... " << std::flush;

  HttpResponse resp = httpGet("/nonexistent");
  CHECK(resp.ok, "not_found: got response");
  CHECK(resp.statusCode == 404, "not_found: status 404");

  std::cout << std::endl;
}

static void testPredictBadJson()
{
  std::cout << "  testPredictBadJson... " << std::flush;

  HttpResponse resp = httpPostJson("/predict", "not json");
  CHECK(resp.ok, "bad_json: got response");
  CHECK(resp.statusCode == 400, "bad_json: status 400");

  nlohmann::json body = nlohmann::json::parse(resp.body.toStdString());
  CHECK(body.contains("error"), "bad_json: has error field");

  std::cout << std::endl;
}

static void testPredictMissingInput()
{
  std::cout << "  testPredictMissingInput... " << std::flush;

  HttpResponse resp = httpPostJson("/predict", R"({"foo":"bar"})");
  CHECK(resp.ok, "missing_input: got response");
  CHECK(resp.statusCode == 400, "missing_input: status 400");

  std::cout << std::endl;
}

static void testPredictBodyTooLarge()
{
  std::cout << "  testPredictBodyTooLarge... " << std::flush;

  // Send a request with Content-Length exceeding MAX_BODY_SIZE_BYTES.
  // Include a small body so the server's waitForReadyRead triggers.
  qint64 fakeSize = MAX_BODY_SIZE_BYTES + 1024;
  QByteArray smallBody(64, 'X');
  QByteArray req = "POST /predict HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Content-Type: image/jpeg\r\n"
                   "Content-Length: " +
                   QByteArray::number(fakeSize) +
                   "\r\n"
                   "Connection: close\r\n"
                   "\r\n" +
                   smallBody;

  // Server should reject based on Content-Length before reading the full body
  HttpResponse resp = sendHttpRequest(req, 10000);
  CHECK(resp.ok, "body_too_large: got response");
  CHECK(resp.statusCode == 413, "body_too_large: status 413");

  if (resp.ok && !resp.body.isEmpty()) {
    nlohmann::json body = nlohmann::json::parse(resp.body.toStdString());
    CHECK(body.contains("error"), "body_too_large: has error field");
  }

  std::cout << std::endl;
}

static void testPredictBodyJustUnderLimit()
{
  std::cout << "  testPredictBodyJustUnderLimit... " << std::flush;

  // Send a request with Content-Length just under the limit (1 MB - 1 KB).
  // This verifies the config value is interpreted as megabytes:
  // if it were bytes, maxBodySize=1 would reject almost everything.
  qint64 justUnder = MAX_BODY_SIZE_BYTES - 1024;
  QByteArray smallBody(64, 'X');
  QByteArray req = "POST /predict HTTP/1.1\r\n"
                   "Host: 127.0.0.1\r\n"
                   "Content-Type: image/jpeg\r\n"
                   "Content-Length: " +
                   QByteArray::number(justUnder) +
                   "\r\n"
                   "Connection: close\r\n"
                   "\r\n" +
                   smallBody;

  // Server should NOT reject — Content-Length is within the limit.
  // It will eventually fail to decode the image, but that's a 500, not 413.
  HttpResponse resp = sendHttpRequest(req, 10000);
  CHECK(resp.ok, "body_under_limit: got response");
  CHECK(resp.statusCode != 413,
        "body_under_limit: not rejected as too large (got " + std::to_string(resp.statusCode) + ")");

  std::cout << std::endl;
}

static void testPredictImageSingle()
{
  std::cout << "  testPredictImageSingle... " << std::flush;

  QByteArray imgData = readImageFile(imagePath("ISIC_4671410.jpg"));
  CHECK(!imgData.isEmpty(), "image_single: loaded test image");

  if (imgData.isEmpty()) {
    std::cout << std::endl;
    return;
  }

  HttpResponse resp = httpPostImage("/predict", imgData);
  CHECK(resp.ok, "image_single: got response");
  CHECK(resp.statusCode == 200, "image_single: status 200");

  nlohmann::json body = nlohmann::json::parse(resp.body.toStdString());
  CHECK(body.contains("output") && body["output"].is_array(), "image_single: has output array");

  auto output = body["output"].get<std::vector<float>>();
  CHECK(static_cast<int>(output.size()) == NUM_OUTPUT, "image_single: output length == " + std::to_string(NUM_OUTPUT));

  // Softmax outputs should sum to ~1.0
  float total = 0.0f;

  for (float v : output)
    total += v;
  CHECK_NEAR(total, 1.0f, 0.01f, "image_single: softmax sums to ~1.0");

  std::cout << std::endl;
}

static void testPredictImageConcurrent()
{
  std::cout << "  testPredictImageConcurrent (5 different images)... " << std::flush;

  // Load all 5 test images
  QStringList imageNames = {"ISIC_1498519.jpg", "ISIC_2729538.jpg", "ISIC_3904045.jpg", "ISIC_4671410.jpg",
                            "ISIC_5186409.jpg"};

  std::vector<QByteArray> imageDataVec;

  for (const QString& name : imageNames) {
    QByteArray data = readImageFile(imagePath(name));
    CHECK(!data.isEmpty(), "concurrent: loaded " + name.toStdString());
    imageDataVec.push_back(data);
  }

  // Send all 5 requests concurrently using std::thread
  std::vector<HttpResponse> responses(imageNames.size());
  std::vector<std::thread> threads;

  for (int i = 0; i < imageNames.size(); i++) {
    threads.emplace_back([&, i]() { responses[i] = httpPostImage("/predict", imageDataVec[i]); });
  }

  for (auto& t : threads)
    t.join();

  // Validate all responses
  for (int i = 0; i < imageNames.size(); i++) {
    std::string tag = "concurrent[" + imageNames[i].toStdString() + "]";
    CHECK(responses[i].ok, tag + ": got response");
    CHECK(responses[i].statusCode == 200, tag + ": status 200");

    nlohmann::json body = nlohmann::json::parse(responses[i].body.toStdString());
    auto output = body["output"].get<std::vector<float>>();
    CHECK(static_cast<int>(output.size()) == NUM_OUTPUT, tag + ": output length");

    float total = 0.0f;

    for (float v : output)
      total += v;
    CHECK_NEAR(total, 1.0f, 0.01f, tag + ": softmax sums to ~1.0");
  }

  std::cout << std::endl;
}

static void testPredictImageRepeatedConcurrent()
{
  std::cout << "  testPredictImageRepeatedConcurrent (10x same image)... " << std::flush;

  QByteArray imgData = readImageFile(imagePath("ISIC_4671410.jpg"));
  CHECK(!imgData.isEmpty(), "repeated: loaded test image");

  if (imgData.isEmpty()) {
    std::cout << std::endl;
    return;
  }

  constexpr int NUM_REQUESTS = 10;
  std::vector<HttpResponse> responses(NUM_REQUESTS);
  std::vector<std::thread> threads;

  for (int i = 0; i < NUM_REQUESTS; i++) {
    threads.emplace_back([&, i]() { responses[i] = httpPostImage("/predict", imgData); });
  }

  for (auto& t : threads)
    t.join();

  // All should succeed
  bool allOk = true;

  for (int i = 0; i < NUM_REQUESTS; i++) {
    if (!responses[i].ok || responses[i].statusCode != 200) {
      allOk = false;
      break;
    }
  }

  CHECK(allOk, "repeated: all " + std::to_string(NUM_REQUESTS) + " responses status 200");

  // All outputs should be identical (deterministic — same input → same output)
  if (allOk) {
    std::string firstBody = responses[0].body.toStdString();
    bool allIdentical = true;

    for (int i = 1; i < NUM_REQUESTS; i++) {
      if (responses[i].body.toStdString() != firstBody) {
        allIdentical = false;
        break;
      }
    }

    CHECK(allIdentical, "repeated: all outputs identical (deterministic)");
  }

  std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void runEndpointTests()
{
  std::cout << "Starting NN-Server..." << std::endl;

  if (!startServer()) {
    std::cerr << "FATAL: Could not start NN-Server. Skipping all endpoint tests." << std::endl;
    testsFailed++;
    return;
  }

  testHealth();
  testNotFound();
  testPredictBadJson();
  testPredictMissingInput();
  testPredictBodyTooLarge();
  testPredictBodyJustUnderLimit();
  testPredictImageSingle();
  testPredictImageConcurrent();
  testPredictImageRepeatedConcurrent();

  stopServer();
}
