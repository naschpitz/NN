#ifndef NN_CLI_APP_HPP
#define NN_CLI_APP_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_LogLevel.hpp"
#include "NN-CLI_NetworkType.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <QCommandLineParser>

#include <memory>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  /**
   * Application entry point and Controller factory for the MVC architecture.
   *
   * Handles initial config loading, network type detection, and delegates
   * to the appropriate Controller (TrainingController, PredictController,
   * or TestController) based on the selected mode.  Each Controller bridges
   * a concrete Runner (Model) and a View through the IRunnerObserver
   * interface.
   *
   * Architecture:
   *   App -> Controller -> {Model (Runner), View (Window)}
   *   Model -> Controller (via Observer notifications)
   *   Controller -> View (via API updates)
   *
   * Calibrate mode is a special case that bypasses the Controller layer
   * and invokes CalibrateRunner directly (it is a CLI-level mode, not a
   * model-level one).
   */
  class App
  {
    public:
      App(const QCommandLineParser& parser, LogLevel logLevel);
      int run();

    private:
      //-- Configuration --//
      const QCommandLineParser& parser;
      LogLevel logLevel;
      NetworkType networkType;
      std::string mode; // "train", "test", "predict"
      bool isCalibrateMode = false; // true when --mode calibrate; runs as predict internally
      IOConfig ioConfig;
      AugmentationConfig augConfig;

      //-- ANN members --//
      std::unique_ptr<ANN::Core<float>> annCore;
      ANN::CoreConfig<float> annCoreConfig;

      //-- CNN members --//
      std::unique_ptr<CNN::Core<float>> cnnCore;
      CNN::CoreConfig<float> cnnCoreConfig;
  };

} // namespace NN_CLI

#endif // NN_CLI_APP_HPP
