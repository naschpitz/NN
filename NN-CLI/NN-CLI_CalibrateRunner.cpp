#include "NN-CLI_CalibrateRunner.hpp"

#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_CNNRunner.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Utils.hpp"

#include "Common/Common_Utils.hpp"

#include <json.hpp>

#include <QFile>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

NN_CLI::CalibrateRunner::CalibrateRunner(const QCommandLineParser& parser, LogLevel logLevel, NetworkType networkType,
                                         const IOConfig& ioConfig, const AugmentationConfig& augConfig,
                                         std::unique_ptr<ANN::Core<float>>& annCore,
                                         const ANN::CoreConfig<float>& annCoreConfig,
                                         std::unique_ptr<CNN::Core<float>>& cnnCore,
                                         const CNN::CoreConfig<float>& cnnCoreConfig)
  : parser(parser),
    logLevel(logLevel),
    networkType(networkType),
    ioConfig(ioConfig),
    augConfig(augConfig),
    annCore(annCore),
    annCoreConfig(annCoreConfig),
    cnnCore(cnnCore),
    cnnCoreConfig(cnnCoreConfig)
{
}

//===================================================================================================================//
//-- Observer management --//
//===================================================================================================================//

void NN_CLI::CalibrateRunner::addObserver(NN_CLI::IRunnerObserver* observer)
{
  if (observer == nullptr)
    return;

  for (auto* existing : this->observers) {
    if (existing == observer)
      return;
  }

  this->observers.push_back(observer);
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::removeObserver(NN_CLI::IRunnerObserver* observer)
{
  if (observer == nullptr)
    return;

  auto it = std::find(this->observers.begin(), this->observers.end(), observer);

  if (it != this->observers.end())
    this->observers.erase(it);
}

//===================================================================================================================//
//-- Observer notifications --//
//===================================================================================================================//

void NN_CLI::CalibrateRunner::notifyLogMessage(const std::string& message, bool isError)
{
  for (auto* observer : this->observers)
    observer->onLogMessage(message, isError);
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::notifyTrainingFinished(bool success, const std::string& finalSummary)
{
  for (auto* observer : this->observers)
    observer->onTrainingFinished(success, finalSummary);
}

//===================================================================================================================//
//-- run --//
//===================================================================================================================//

int NN_CLI::CalibrateRunner::run()
{
  // ---- args --------------------------------------------------------------
  if (!this->parser.isSet("id-images")) {
    std::string errMsg = "Error: --id-images <dir> is required for calibrate mode.";
    std::cerr << errMsg << "\n";
    this->notifyLogMessage(errMsg, true);
    return 1;
  }

  // ---- merge CLI args into coreConfig.calibrationConfig ---------------------
  this->annCoreConfig.calibrationConfig.idImagesDir = this->parser.value("id-images").toStdString();
  this->annCoreConfig.calibrationConfig.oodDir = this->parser.isSet("ood-dir")
                                                   ? this->parser.value("ood-dir").toStdString()
                                                   : (fs::current_path() / "extern-datasets" / "ood").string();
  this->annCoreConfig.calibrationConfig.idSampleCount = this->parser.isSet("id-sample-count")
                                                           ? this->parser.value("id-sample-count").toULongLong()
                                                           : 500;
  this->annCoreConfig.calibrationConfig.oodSampleCount = this->parser.isSet("ood-sample-count")
                                                           ? this->parser.value("ood-sample-count").toULongLong()
                                                           : 1500;
  this->annCoreConfig.calibrationConfig.idPercentile = this->parser.isSet("id-percentile")
                                                          ? this->parser.value("id-percentile").toDouble()
                                                          : 95.0;
  this->annCoreConfig.calibrationConfig.fetchIfMissing = !this->parser.isSet("no-fetch");
  if (this->parser.isSet("output")) {
    this->annCoreConfig.calibrationConfig.outputPath = this->parser.value("output").toStdString();
  } else {
    fs::path configPath = this->parser.value("config").toStdString();
    this->annCoreConfig.calibrationConfig.outputPath =
        (configPath.parent_path() / "threshold.json").string();
  }
  this->annCoreConfig.calibrationConfig.logLevel = static_cast<Common::LogLevel>(this->logLevel);
  this->annCoreConfig.calibrationConfig.progressReports = this->ioConfig.progressReports;

  if (this->logLevel > LogLevel::QUIET) {
    std::string msg = "Calibrate mode\n"
                      "  Model:           " +
                      this->parser.value("config").toStdString() +
                      "\n"
                      "  ID images:       " +
                      this->annCoreConfig.calibrationConfig.idImagesDir +
                      "  (sample " + std::to_string(this->annCoreConfig.calibrationConfig.idSampleCount) +
                      ")\n"
                      "  OOD dir:         " +
                      this->annCoreConfig.calibrationConfig.oodDir +
                      "  (sample " + std::to_string(this->annCoreConfig.calibrationConfig.oodSampleCount) +
                      ")\n"
                      "  ID percentile:   " +
                      std::to_string(this->annCoreConfig.calibrationConfig.idPercentile) +
                      "\n"
                      "  Output:          " +
                      this->annCoreConfig.calibrationConfig.outputPath + "\n\n";
    std::cout << msg;
    this->notifyLogMessage(msg, false);
  }

  // ---- create temp runner and delegate -------------------------------------
  if (this->networkType == NetworkType::CNN) {
    // Mirror ANN calibration config to CNN config (same params for both arch types)
    this->cnnCoreConfig.calibrationConfig = this->annCoreConfig.calibrationConfig;

    CNNRunner runner(this->parser, this->logLevel, this->ioConfig, this->augConfig, this->cnnCore,
                     this->cnnCoreConfig);
    runner.addObserver(this);
    int rc = runner.calibrate();
    runner.removeObserver(this);
    return rc;
  } else {
    ANNRunner runner(this->parser, this->logLevel, this->ioConfig, this->augConfig, this->annCore,
                     this->annCoreConfig);
    runner.addObserver(this);
    int rc = runner.calibrate();
    runner.removeObserver(this);
    return rc;
  }
}

//===================================================================================================================//
//-- IRunnerObserver overrides (no-op) --//
//===================================================================================================================//

void NN_CLI::CalibrateRunner::onBatchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                              float etaSeconds, const std::vector<float>& fractions)
{
  (void)batchIdx;
  (void)totalBatches;
  (void)currentLoss;
  (void)samplesPerSec;
  (void)etaSeconds;
  (void)fractions;
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::onEpochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss,
                                               float valLoss, const std::string& summary)
{
  (void)epochIdx;
  (void)totalEpochs;
  (void)epochLoss;
  (void)hasValLoss;
  (void)valLoss;
  (void)summary;
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::onTrainingFinished(bool success, const std::string& finalSummary)
{
  (void)success;
  (void)finalSummary;
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::onModelInfoUpdated(const std::string& property, const std::string& value)
{
  (void)property;
  (void)value;
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::onLogMessage(const std::string& message, bool isError)
{
  (void)message;
  (void)isError;
}

//===================================================================================================================//

void NN_CLI::CalibrateRunner::onTimingUpdated(const std::string& metric, float value)
{
  (void)metric;
  (void)value;
}
