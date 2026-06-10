#include "NN-CLI_TrainingTui.hpp"

#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI.hpp"

#include <OCLW_Core.hpp>

#include <algorithm>
#include <mutex>

namespace NN_CLI
{

  void TrainingTui::attach(std::shared_ptr<TerminalUI> tui, std::function<void()> onResize)
  {
    this->tui = std::move(tui);

    if (!this->tui)
      return;

    if (onResize)
      this->tui->setResizeCallback(std::move(onResize));

    // After a resize redraws the panels, layout() has erased the loading sub-window. Repaint the
    // loading bar from the last-known state so it survives the resize instead of staying blank
    // until the next mini-batch load callback fires.
    this->tui->setOverlayCallback([this]() { this->renderBar(); });
  }

  void TrainingTui::resolveBarGpus(bool deviceIsGpu, int numGpusConfig)
  {
    this->barGpus = 1;

    if (!deviceIsGpu)
      return;

    OpenCLWrapper::Core::initialize(false);
    int availableGpus = static_cast<int>(OpenCLWrapper::Core::getNumDevices());
    this->barGpus = (numGpusConfig > 0) ? std::min(availableGpus, numGpusConfig) : availableGpus;
    this->barGpus = std::max(1, this->barGpus);
  }

  std::function<void(ulong, ulong, ulong, ulong, SampleLoadType)> TrainingTui::loadingCallback()
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

  void TrainingTui::renderBar()
  {
    if (!this->loading || !this->tui)
      return;

    ProgressBar::renderLoadingBar(this->tui->loadingWindow(), this->current, this->total, this->batchNum,
                                  this->totalBatches, this->barGpus);
  }

} // namespace NN_CLI
