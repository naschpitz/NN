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
    this->tui_ = std::move(tui);

    if (!this->tui_)
      return;

    if (onResize)
      this->tui_->setResizeCallback(std::move(onResize));

    // After a resize redraws the panels, layout() has erased the loading sub-window. Repaint the
    // loading bar from the last-known state so it survives the resize instead of staying blank
    // until the next mini-batch load callback fires.
    this->tui_->setOverlayCallback([this]() { this->renderBar(); });
  }

  void TrainingTui::resolveBarGpus(bool deviceIsGpu, int numGpusConfig)
  {
    this->barGpus_ = 1;

    if (!deviceIsGpu)
      return;

    OpenCLWrapper::Core::initialize(false);
    int availableGpus = static_cast<int>(OpenCLWrapper::Core::getNumDevices());
    this->barGpus_ = (numGpusConfig > 0) ? std::min(availableGpus, numGpusConfig) : availableGpus;
    this->barGpus_ = std::max(1, this->barGpus_);
  }

  std::function<void(ulong, ulong, ulong, ulong)> TrainingTui::loadingCallback()
  {
    return [this](ulong current, ulong total, ulong batchNum, ulong totalBatches) {
      this->current_ = current;
      this->total_ = total;
      this->batchNum_ = batchNum;
      this->totalBatches_ = totalBatches;
      this->loading_ = true;

      std::lock_guard<std::recursive_mutex> lock(this->tui_->mutex());
      this->tui_->handleResize();
      this->renderBar();
    };
  }

  void TrainingTui::renderBar()
  {
    if (!this->loading_ || !this->tui_)
      return;

    ProgressBar::renderLoadingBar(this->tui_->loadingWindow(), this->current_, this->total_, this->batchNum_,
                                  this->totalBatches_, this->barGpus_);
  }

} // namespace NN_CLI
