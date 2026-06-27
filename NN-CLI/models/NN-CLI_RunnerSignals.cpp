#include "NN-CLI_RunnerSignals.hpp"

namespace NN_CLI
{

  //===================================================================================================================//

  void connectRunnerSignals(RunnerSignals& hub, QObject* context, IRunnerObserver* observer)
  {
    qRegisterMetaType<std::string>();
    qRegisterMetaType<std::vector<float>>();
    qRegisterMetaType<Common::PredictResults<float>>();

    QObject::connect(&hub, &RunnerSignals::sampleLoadProgress, context,
                     [observer](ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation) {
                       observer->onSampleLoadProgress(current, total, batchIndex, totalBatches, isValidation);
                     });

    QObject::connect(&hub, &RunnerSignals::validationProgress, context,
                     [observer](ulong current, ulong total) { observer->onValidationProgress(current, total); });

    QObject::connect(&hub, &RunnerSignals::batchProgress, context,
                     [observer](int batchIdx, int totalBatches, float currentLoss, float samplesPerSec,
                                float etaSeconds, const std::vector<float>& fractions) {
                       observer->onBatchProgress(batchIdx, totalBatches, currentLoss, samplesPerSec, etaSeconds,
                                                 fractions);
                     });

    QObject::connect(&hub, &RunnerSignals::epochCompleted, context,
                     [observer](int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                                float learningRate, const std::string& summary) {
                       observer->onEpochCompleted(epochIdx, totalEpochs, epochLoss, hasValLoss, valLoss, learningRate,
                                                  summary);
                     });

    QObject::connect(
      &hub, &RunnerSignals::trainFinished, context,
      [observer](bool success, const std::string& finalSummary) { observer->onTrainFinished(success, finalSummary); });

    QObject::connect(&hub, &RunnerSignals::predictFinished, context,
                     [observer](const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                                const std::string& durationFormatted, const std::string& outputPath) {
                       observer->onPredictFinished(results, numInputs, durationSeconds, durationFormatted, outputPath);
                     });

    QObject::connect(&hub, &RunnerSignals::modelInfoUpdated, context,
                     [observer](const std::string& property, const std::string& value) {
                       observer->onModelInfoUpdated(property, value);
                     });

    QObject::connect(&hub, &RunnerSignals::logMessage, context, [observer](const std::string& message, bool isError) {
      observer->onLogMessage(message, isError);
    });

    QObject::connect(&hub, &RunnerSignals::timingUpdated, context,
                     [observer](const std::string& metric, float value) { observer->onTimingUpdated(metric, value); });
  }

} // namespace NN_CLI
