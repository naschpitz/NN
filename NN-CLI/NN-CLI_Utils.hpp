#ifndef NN_CLI_UTILS_HPP
#define NN_CLI_UTILS_HPP

#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI.hpp"

#include <ANN_Core.hpp>
#include <CNN_Types.hpp>
#include <CNN_Sample.hpp>

#include <QCommandLineParser>
#include <QDir>
#include <QFileInfo>
#include <QString>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  //-------------------------------------------------------------------------------------------------------------------//
  //  ValidationState — shared between ANN and CNN runners
  //-------------------------------------------------------------------------------------------------------------------//

  struct ValidationState {
      bool enabled = false;
      ulong checkInterval = 1;
      ulong numValSamples = 0;
      float bestValLoss = std::numeric_limits<float>::max();
      ulong bestValEpoch = 0;
      float lastValLoss = 0.0f;
  };

  //-------------------------------------------------------------------------------------------------------------------//

  inline QString ensureOutputDir(const QString& outputPath)
  {
    QDir outputDir(outputPath);

    if (!outputDir.exists()) {
      outputDir.mkpath(".");
    }

    return outputDir.absolutePath();
  }

  //-------------------------------------------------------------------------------------------------------------------//
  //  Shared CLI helpers — progress bars, validation guards
  //-------------------------------------------------------------------------------------------------------------------//

  /// Set up a test/predict-mode progress callback on the core.
  /// Prints an initial "0 / total" bar and wires a callback that prints progress
  /// on every batch completion.
  template <typename CoreType>
  inline void setupModeProgressCallback(CoreType& core, LogLevel logLevel, ulong progressReports,
                                        const std::string& label, ulong total)
  {
    if (logLevel > LogLevel::QUIET) {
      ProgressBar::printLoadingProgress(label, 0, total, progressReports);
      core.setProgressCallback([progressReports, label](ulong current, ulong total_) {
        ProgressBar::printLoadingProgress(label, current, total_, progressReports);
      });
    }
  }

  //-------------------------------------------------------------------------------------------------------------------//

  /// Compute inverse-frequency class weights from one-hot output vectors.
  /// Shared by both ANN and CNN runners.
  inline std::vector<float> computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs)
  {
    if (outputs.empty())
      return {};

    ulong numClasses = outputs[0].size();
    std::vector<ulong> classCounts(numClasses, 0);

    for (const auto& output : outputs) {
      ulong cls = static_cast<ulong>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));

      if (cls < numClasses)
        classCounts[cls]++;
    }

    ulong totalSamples = outputs.size();
    std::vector<float> weights(numClasses, 1.0f);

    for (ulong c = 0; c < numClasses; c++) {
      if (classCounts[c] > 0) {
        weights[c] =
          static_cast<float>(totalSamples) / (static_cast<float>(numClasses) * static_cast<float>(classCounts[c]));
      }
    }

    return weights;
  }

  //-------------------------------------------------------------------------------------------------------------------//

  /// Check whether both --samples and --idx-data were passed.  Prints an error
  /// message and returns true when they conflict.
  inline bool checkSamplesIdxDataConflict(const QCommandLineParser& parser)
  {
    if (parser.isSet("samples") && parser.isSet("idx-data")) {
      std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
      return true;
    }

    return false;
  }

  //-------------------------------------------------------------------------------------------------------------------//

  /// Attach a validation progress bar callback to a validation core so the TUI
  /// progress window shows validation progress.
  template <typename CoreType>
  inline void setupValidationProgressCallback(CoreType& validationCore, std::shared_ptr<TerminalUI> tui,
                                              ulong validationTotal, int totalGPUs)
  {
    if (tui && tui->isInitialized()) {
      int validationGpus = std::max(1, totalGPUs);
      validationCore.setProgressCallback([tui, validationTotal, validationGpus](ulong current, ulong) {
        float pct = static_cast<float>(current) / static_cast<float>(validationTotal) * 100.0f;
        std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());
        tui->handleResize();
        ProgressBar::renderValidationBar(tui->progressWindow(), pct, validationGpus);
      });
    }
  }

  //-------------------------------------------------------------------------------------------------------------------//

  template <typename T>
  class Utils
  {
    public:
      /// Load IDX dataset as ANN samples (flat input vectors)
      static ANN::Samples<T> loadANNIDX(const std::string& dataPath, const std::string& labelsPath,
                                        ulong progressReports = 1000);

      /// Load IDX dataset as CNN samples (3D tensor inputs with given shape)
      static CNN::Samples<T> loadCNNIDX(const std::string& dataPath, const std::string& labelsPath,
                                        const CNN::Shape3D& inputShape, ulong progressReports = 1000);

    private:
      static uint32_t readBigEndianUInt32(std::ifstream& stream);
      static std::vector<std::vector<unsigned char>> loadIDXData(const std::string& path);
      static std::vector<unsigned char> loadIDXLabels(const std::string& path);
  };

} // namespace NN_CLI

#endif // NN_CLI_UTILS_HPP
