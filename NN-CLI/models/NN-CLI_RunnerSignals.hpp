#ifndef NN_CLI_RUNNERSIGNALS_HPP
#define NN_CLI_RUNNERSIGNALS_HPP

#include "Common/Common_PredictResult.hpp"
#include "NN-CLI_Types.hpp"

#include <QObject>

#include <string>
#include <vector>

// Metatype declarations for queued-connection argument types.  Without these,
// Qt cannot deep-copy the arguments when a signal is emitted cross-thread
// (AutoConnection → QueuedConnection).
Q_DECLARE_METATYPE(std::string)
Q_DECLARE_METATYPE(std::vector<float>)
Q_DECLARE_METATYPE(Common::PredictResults<float>)

//===================================================================================================================//

namespace NN_CLI
{
  // Forward declaration for friendship (Runner emits these signals directly).
  template <typename CoreT, typename CoreConfigT>
  class Runner;

  // Non-template QObject "signals hub" owned by Runner<CoreT, CoreConfigT>.
  // Runner is a class template and moc cannot process Q_OBJECT in a template,
  // so the signal surface lives in this non-template class.  Runner holds a
  // RunnerSignals member and emits through it.
  //
  // Connections use default AutoConnection: cross-thread emits (from Core
  // worker threads) are queued and delivered on the receiver's thread.
  // Custom argument types are registered via qRegisterMetaType in the
  // constructor.
  class RunnerSignals : public QObject
  {
      Q_OBJECT

    public:
      //-- Constructors --//
      explicit RunnerSignals(QObject* parent = nullptr) : QObject(parent)
      {
        // Register custom types so queued (cross-thread) connections can
        // deep-copy the arguments.  Idempotent and safe to call once per hub.
        qRegisterMetaType<std::string>();
        qRegisterMetaType<std::vector<float>>();
        qRegisterMetaType<Common::PredictResults<float>>();
      }

    signals:
      //-- Data-loading progress (per-batch sample load) --//
      void sampleLoadProgress(ulong current, ulong total, ulong batchIndex, ulong totalBatches, bool isValidation);

      //-- Validation-pass progress (epoch boundary) --//
      void validationProgress(ulong current, ulong total);

      //-- Per mini-batch training progress --//
      void batchProgress(int batchIdx, int totalBatches, float currentLoss, float samplesPerSec, float etaSeconds,
                         const std::vector<float>& fractions);

      //-- Epoch completion --//
      void epochCompleted(int epochIdx, int totalEpochs, float epochLoss, bool hasValLoss, float valLoss,
                          float learningRate, const std::string& summary);

      //-- Training run finished --//
      void trainFinished(bool success, const std::string& finalSummary);

      //-- Prediction finished --//
      void predictFinished(const Common::PredictResults<float>& results, size_t numInputs, double durationSeconds,
                           const std::string& durationFormatted, const std::string& outputPath);

      //-- Model property updated --//
      void modelInfoUpdated(const std::string& property, const std::string& value);

      //-- General-purpose log message --//
      void logMessage(const std::string& message, bool isError);

      //-- Timing/profiling metric updated --//
      void timingUpdated(const std::string& metric, float value);

    private:
      //-- Runner emits the signals directly (signals are protected, hence friendship) --//
      template <typename CoreT, typename CoreConfigT>
      friend class Runner;
  };

  //===================================================================================================================//

} // namespace NN_CLI

#endif // NN_CLI_RUNNERSIGNALS_HPP
