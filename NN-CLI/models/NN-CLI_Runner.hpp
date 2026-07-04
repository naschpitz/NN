#ifndef NN_CLI_RUNNER_HPP
#define NN_CLI_RUNNER_HPP

#include "NN-CLI_RunnerBase.hpp"

#include <memory>

//===================================================================================================================//

namespace NN_CLI
{

  /**
   * Template Runner providing the typed implementations of RunnerBase's
   * virtual methods.  Controllers interact only through RunnerBase.
   */
  template <typename CoreT, typename CoreConfigT>
  class Runner : public RunnerBase
  {
    public:
      //-- Constructors --//

      Runner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig, AugmentationConfig& augConfig,
             std::unique_ptr<CoreT>& core, CoreConfigT& coreConfig, const QString& configPath);

      //-- Abort --//

      void requestAbort() override
      {
        if (this->core)
          this->core->requestStop();
      }

      //-- Typed accessor (template-only, internal use) --//

      const CoreConfigT& getCoreConfig() const
      {
        return this->coreConfig;
      }

      //-- Virtual accessor overrides --//

      int getTotalEpochs() const override;

      std::string getClassWeightsString() const override;

      std::string getLearningRateSchedulerString() const override;

      std::vector<SummaryRow> buildModelInfoRows() const override;

      std::vector<SummaryRow> buildPredictModelInfoRows() const override;

      const Common::TrainMetadata<float>& getLoadedTrainMetadata() const override
      {
        return this->coreConfig.loadedTrainMetadata;
      }

    protected:
      //-- Shared helpers (typed implementations) --//

      int finishTrain(const QString& inputFilePath) override;

      void applyLearningRateScheduler(ulong epoch, int totalEpochs, bool hasValLoss, float valLoss) override;

      // Connect a core's predictProgress signal to emit this->batchProgress, throttled by
      // progressReports.  Wired once in the ctor for this->core (so a reused Runner never stacks
      // duplicate connections); also called from test() on its local class-weighted testCore,
      // which is a fresh object destroyed at the end of each test() run.  The core reports its
      // own per-phase total, so a single connection serves multi-phase modes like calibrate.
      void emitProgressFromCore(CoreT& core);

      void doSaveModel(const std::string& outputPath) override = 0;

      //-- Typed state --//

      std::unique_ptr<CoreT>& core;
      CoreConfigT& coreConfig;
  };

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_RUNNER_HPP
