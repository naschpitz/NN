#include "NN-CLI_TrainingController.hpp"

#include "NN-CLI_TerminalUI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI.hpp"

#include <OCLW_Core.hpp>

#include <algorithm>
#include <mutex>
#include <sstream>

namespace NN_CLI
{

  void TrainingController::attach(std::shared_ptr<TerminalUI> tui, std::function<void()> onResize)
  {
    this->tui = std::move(tui);

    if (!this->tui)
      return;

    if (onResize)
      this->tui->setResizeCallback(std::move(onResize));

    // After a resize redraws the panels, layout() has erased the progress sub-window. Repaint
    // the loading bar from the last-known state so it survives the resize instead of staying blank
    // until the next mini-batch load callback fires.
    this->tui->setOverlayCallback([this]() { this->renderBar(); });
  }

  void TrainingController::resolveBarGpus(bool deviceIsGpu, int numGpusConfig)
  {
    this->barGpus = 1;

    if (!deviceIsGpu)
      return;

    OpenCLWrapper::Core::initialize(false);
    int availableGpus = static_cast<int>(OpenCLWrapper::Core::getNumDevices());
    this->barGpus = (numGpusConfig > 0) ? std::min(availableGpus, numGpusConfig) : availableGpus;
    this->barGpus = std::max(1, this->barGpus);
  }

  std::function<void(ulong, ulong, ulong, ulong, SampleLoadType)> TrainingController::loadingCallback()
  {
    return [this](ulong current, ulong total, ulong batchNum, ulong totalBatches, SampleLoadType loadType) {
      if (loadType != SampleLoadType::Training)
        return; // only track training sample loading

      this->current = current;
      this->total = total;
      this->batchNum = batchNum;
      this->totalBatches = totalBatches;
      this->loading = true;

      std::lock_guard<std::recursive_mutex> lock(this->tui->getMutex());
      this->tui->handleResize();
      this->renderBar();
    };
  }

  //===================================================================================================================//

  void TrainingController::initTracker(ulong progressReports, ulong windowSize, int barWidth)
  {
    this->progressTracker = std::make_unique<TrainingProgressTracker>(progressReports, windowSize, barWidth);
  }

  //===================================================================================================================//

  void TrainingController::updateProgress(const ProgressInfo& progress)
  {
    if (!this->progressTracker)
      return;

    if (!this->tui || !this->tui->isInitialized())
      return;

    this->progressTracker->update(progress, this->tui->progressWindow());
  }

  //===================================================================================================================//

  void TrainingController::renderBar()
  {
    if (!this->loading || !this->tui)
      return;

    std::ostringstream labelStream;
    labelStream << "Samples (" << this->batchNum << "/" << this->totalBatches << ")";
    std::string label = labelStream.str();
    float fraction = (this->total > 0) ? static_cast<float>(this->current) / static_cast<float>(this->total) : 0.0f;

    TerminalUI_ProgressBar renderer;
    renderer.renderSingleBar(this->tui->progressWindow(), label, fraction);
  }

} // namespace NN_CLI
