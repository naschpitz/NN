#ifndef NN_CLI_RUNNERBASE_HPP
#define NN_CLI_RUNNERBASE_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_Utils.hpp"

#include "Common/Common_LearningRateScheduler.hpp"
#include "Common/Common_PredictResult.hpp"
#include "Common/Common_TrainMetadata.hpp"
#include "Common/Common_TrainProgressEvent.hpp"

#include <QCommandLineParser>
#include <QMutex>

#include <chrono>
#include <deque>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // Non-template QObject base for Runner.  Provides the signal interface and
  // all shared state/methods that do not depend on the typed Core/CoreConfig
  // template parameters.  Controllers interact exclusively through this base,
  // enabling them to be non-template QObjects with real Qt slots connected
  // directly to these signals.
  //
  // The template Runner<CoreT, CoreConfigT> inherits RunnerBase and supplies
  // the typed implementations of the pure-virtual methods.
  class RunnerBase : public QObject
  {
      Q_OBJECT

    public:
      //-- Mode methods (implemented by the template Runner) --//

      virtual int train() = 0;
      virtual int predict() = 0;
      virtual int test() = 0;
      virtual int calibrate() = 0;

      virtual ~RunnerBase() = default;

      //-- Abort --//

      virtual void requestAbort() = 0;

      //-- Virtual accessors (access typed core/config in derived Runner) --//

      virtual int getTotalEpochs() const = 0;

      virtual std::string getClassWeightsString() const = 0;

      virtual std::string getLearningRateSchedulerString() const = 0;

      virtual std::vector<SummaryRow> buildModelInfoRows() const = 0;

      virtual std::vector<SummaryRow> buildPredictModelInfoRows() const = 0;

      virtual const Common::TrainMetadata<float>& getLoadedTrainMetadata() const = 0;

      //-- Model info virtual accessors --//

      virtual std::vector<std::string> getTimingLines(int maxWidth = 0) const = 0;

      virtual ulong getNumOutputClasses() const = 0;

      virtual ulong getTotalParameters() const = 0;

      virtual std::string getNetworkType() const = 0;

      virtual std::string getInputShapeString() const = 0;

      virtual ulong getNumConvLayers() const
      {
        return 0;
      }

      virtual ulong getNumDenseLayers() const = 0;

      virtual ulong getNumResidualBlocks() const
      {
        return 0;
      }

      //-- Non-virtual accessors (access shared state only) --//

      float getCurrentLearningRate() const;

      const IOConfig& getIOConfig() const;

      ulong getNumOriginalTrainSamples() const
      {
        return this->_numOriginalTrainSamples;
      }

      ulong getNumTrainSamples() const
      {
        return this->_numTrainSamples;
      }

      ulong getNumValidationSamples() const
      {
        return this->_numValidationSamples;
      }

      std::string getAugmentationString() const;

      std::string getValidationString() const;

    signals:
      //-- Runner signals (connected directly to Controller slots) --//

      void sampleLoadProgress(ulong current, ulong count, ulong batchIndex, ulong totalBatches, bool isValidation);

      void validationProgress(ulong current, ulong total);

      void batchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                         const std::vector<float>& fractions);

      void epochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                          float learningRate, const std::string& summary);

      void trainFinished(bool success, const std::string& finalSummary);

      void predictFinished(const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                           const std::string& durationFormatted, const std::string& outputPath);

      void modelInfoUpdated(const std::string& property, const std::string& value);

      void logMessage(const std::string& message, bool isError);

      void timingUpdated(const std::string& metric, float value);

    protected:
      //-- Pure virtual shared helpers (access typed core/config) --//

      virtual int finishTrain(const QString& inputFilePath) = 0;

      virtual void applyLearningRateScheduler(ulong epoch, int totalEpochs, bool hasValLoss, float valLoss) = 0;

      virtual void setupPredictProgressCallback(ulong total) = 0;

      virtual void doSaveModel(const std::string& outputPath) = 0;

      //-- Non-virtual shared helpers --//

      void handleTrainProgress(const Common::TrainProgressEvent<float>& progress, ulong batchSize);

      ValidationMetadata buildValidationMetadata() const;

      //-- Constructor --//

      RunnerBase(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
                 const QString& configPath);

      //-- Shared state --//

      const QCommandLineParser& parser;
      LogLevel logLevel;
      QString configPath;
      IOConfig& ioConfig;
      AugmentationConfig& augConfig;

      ulong _numOriginalTrainSamples = 0;
      ulong _numTrainSamples = 0;
      ulong _numValidationSamples = 0;

      ValidationState validationState;

      Common::LearningRateSchedulerState learningRateSchedulerState;

      float lastEpochLoss = 0.0f;
      std::vector<float> gpuFractions;
      int trackedEpoch = -1;
      int trackedTotalGPUs = 0;

      double runningLossSum = 0.0;
      ulong runningLossCount = 0;
      int statsEpoch = -1;
      std::chrono::steady_clock::time_point epochStartTime;

      struct RateSample {
          double samplesDone;
          std::chrono::steady_clock::time_point timestamp;
      };

      std::deque<RateSample> rateWindow;

      QMutex callbackMutex;
  };

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_RUNNERBASE_HPP
