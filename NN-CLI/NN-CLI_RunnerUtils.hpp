#ifndef NN_CLI_RUNNERUTILS_HPP
#define NN_CLI_RUNNERUTILS_HPP

#include "NN-CLI_DataType.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_Utils.hpp"

#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//
  //  finishTrainingCommon
  //===================================================================================================================//

  /// Common finish-training logic shared by ANN and CNN runners.
  /// @param logLevel      Current log level.
  /// @param parser        CLI parser (for --output).
  /// @param inputFilePath Path to the input samples file (used for default output path).
   /// @param core          The trained core (ANN or CNN) — must support getTrainConfig() and getTrainMetadata().
  /// @param saveFn        Callable that takes (const std::string& outputPath) and persists the model.
  template <typename CoreT, typename SaveFn>
  int finishTrainingCommon(LogLevel logLevel, const QCommandLineParser& parser, const QString& inputFilePath,
                           const CoreT& core, const SaveFn& saveFn)
  {
    if (logLevel > LogLevel::QUIET)
      std::cout << "\nTraining completed.\n";

    const auto& trainConfig = core.getTrainConfig();
    const auto& trainMetadata = core.getTrainMetadata();
    // lastEpoch is a 0-based index, so the count of epochs trained is index + 1.
    // Fall back to the configured numEpochs when no epoch completed.
    ulong actualEpochs =
      trainMetadata.epochHistory.empty() ? trainConfig.numEpochs : trainMetadata.epochHistory.back().epoch + 1;

    std::string outputPathStr;

    if (parser.isSet("output")) {
      outputPathStr = parser.value("output").toStdString();
    } else {
      outputPathStr = ModelSerializer::generateDefaultOutputPath(
        inputFilePath, actualEpochs, trainMetadata.numSamples, trainMetadata.finalLoss);
    }

    saveFn(outputPathStr);

    if (logLevel > LogLevel::QUIET)
      std::cout << "Model saved to: " << outputPathStr << "\n";
    return 0;
  }

  //===================================================================================================================//
  //  resolvePredictOutputPath
  //===================================================================================================================//

  /// Resolve the output path for predict mode.  If --output is not set, derives from input path.
  inline QString resolvePredictOutputPath(const QCommandLineParser& parser, const IOConfig& ioConfig)
  {
    QString inputPath = parser.value("input");
    QString outputPath;

    if (parser.isSet("output")) {
      outputPath = parser.value("output");
    } else {
      QFileInfo inputInfo(inputPath);
      QDir inputDir = inputInfo.absoluteDir();
      QDir outputDir(inputDir.filePath("output"));

      if (!outputDir.exists())
        inputDir.mkdir("output");

      if (ioConfig.outputType == DataType::IMAGE) {
        outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
      } else {
        outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
      }
    }

    return outputPath;
  }

  //===================================================================================================================//
  //  writePredictOutput
  //===================================================================================================================//

  /// Write predict results to disk (images or JSON).
  /// @param results            Container of prediction results — each element must have .output and .logits members.
  /// @param outputPath         File/directory path for output.
  /// @param ioConfig           I/O configuration (output type, shape).
  /// @param logLevel           Current log level.
  /// @param startTimeStr       ISO-8601 start timestamp.
  /// @param endTimeStr         ISO-8601 end timestamp.
  /// @param durationSeconds    Elapsed wall time in seconds.
  /// @param durationFormatted  Human-readable duration string.
  /// @param numInputs          Total number of inputs that were predicted.
  template <typename ResultsT>
  int writePredictOutput(const ResultsT& results, const QString& outputPath, const IOConfig& ioConfig,
                         LogLevel logLevel, const std::string& startTimeStr, const std::string& endTimeStr,
                         double durationSeconds, const std::string& durationFormatted, size_t numInputs)
  {
    // When outputType is IMAGE, save images to a folder
    if (ioConfig.outputType == DataType::IMAGE) {
      if (!ioConfig.hasOutputShape()) {
        std::cerr << "Error: outputType is 'image' but no outputShape provided in config.\n";
        return 1;
      }

      QDir outDir(outputPath);

      if (!outDir.exists())
        QDir().mkpath(outputPath);

      for (size_t i = 0; i < results.size(); ++i) {
        QString imgName = QString::number(i) + ".png";
        std::string imgPath = outDir.filePath(imgName).toStdString();
        ImageLoader::saveImage(imgPath, results[i].output, static_cast<int>(ioConfig.outputC),
                               static_cast<int>(ioConfig.outputH), static_cast<int>(ioConfig.outputW));
      }

      if (logLevel > LogLevel::QUIET) {
        std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
        std::cout << "  Images: " << results.size() << "\n";
        std::cout << "  Shape: " << ioConfig.outputC << "x" << ioConfig.outputH << "x" << ioConfig.outputW << "\n";
        std::cout << "  Duration: " << durationFormatted << "\n";
      }

      return 0;
    }

    // Standard vector output: save as JSON.
    // For each input we emit both the post-activation `output` and the pre-activation
    // `logits` of the last layer so callers can compute calibration / OOD scores
    // (max-logit, logit-norm, free-energy) that softmax discards.
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<float>> logits;
    outputs.reserve(results.size());
    logits.reserve(results.size());

    for (const auto& r : results) {
      outputs.push_back(r.output);
      logits.push_back(r.logits);
    }

    nlohmann::ordered_json resultJson;
    nlohmann::ordered_json predictMetadataJson;
    predictMetadataJson["startTime"] = startTimeStr;
    predictMetadataJson["endTime"] = endTimeStr;
    predictMetadataJson["durationSeconds"] = durationSeconds;
    predictMetadataJson["durationFormatted"] = durationFormatted;
    predictMetadataJson["numInputs"] = numInputs;
    resultJson["predictMetadata"] = predictMetadataJson;
    resultJson["outputs"] = outputs;
    resultJson["logits"] = logits;

    QFile outputFile(outputPath);

    if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
      std::cerr << "Error: Failed to open output file: " << outputPath.toStdString() << "\n";
      return 1;
    }

    std::string jsonStr = resultJson.dump(2);
    outputFile.write(jsonStr.c_str(), jsonStr.size());
    outputFile.close();

    if (logLevel > LogLevel::QUIET)
      std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
    return 0;
  }

  //===================================================================================================================//
  //  loadSamplesFromOptionsCommon
  //===================================================================================================================//

  /// Common sample loading logic for ANN and CNN runners.
  /// @param parser             CLI parser (for --samples, --idx-data, --idx-labels).
  /// @param logLevel           Current log level.
  /// @param ioConfig           I/O configuration.
  /// @param modeName           Human-readable mode name ("train", "test").
  /// @param inputFilePath      [out] Set to the resolved file path.
  /// @param loadJsonSamples    Callable: (std::string path, ulong progressReports) → SamplesT
  /// @param loadIdxSamples     Callable: (std::string dataPath, std::string labelsPath, ulong progressReports) → SamplesT
  template <typename SamplesT, typename LoadJsonFn, typename LoadIdxFn>
  std::pair<SamplesT, bool> loadSamplesFromOptionsCommon(const QCommandLineParser& parser, LogLevel logLevel,
                                                         const IOConfig& ioConfig, const std::string& modeName,
                                                         QString& inputFilePath, LoadJsonFn&& loadJsonSamples,
                                                         LoadIdxFn&& loadIdxSamples)
  {
    SamplesT samples;

    bool hasJsonSamples = parser.isSet("samples");
    bool hasIdxData = parser.isSet("idx-data");
    bool hasIdxLabels = parser.isSet("idx-labels");

    if (checkSamplesIdxDataConflict(parser))
      return {samples, false};

    ulong displayProgressReports = (logLevel > LogLevel::QUIET) ? ioConfig.progressReports : 0;

    if (hasJsonSamples) {
      QString samplesPath = parser.value("samples");
      inputFilePath = samplesPath;

      if (logLevel >= LogLevel::INFO)
        std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
      samples = loadJsonSamples(samplesPath.toStdString(), displayProgressReports);
    } else if (hasIdxData) {
      if (!hasIdxLabels) {
        std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
        return {samples, false};
      }

      QString idxDataPath = parser.value("idx-data");
      QString idxLabelsPath = parser.value("idx-labels");
      inputFilePath = idxDataPath;

      if (logLevel >= LogLevel::INFO) {
        std::cout << "Loading " << modeName << " samples from IDX:\n";
        std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
        std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
      }

      samples = loadIdxSamples(idxDataPath.toStdString(), idxLabelsPath.toStdString(), displayProgressReports);
    } else {
      std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
      return {samples, false};
    }

    if (logLevel >= LogLevel::INFO)
      std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

    return {samples, true};
  }

} // namespace NN_CLI

#endif // NN_CLI_RUNNERUTILS_HPP
