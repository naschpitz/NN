#ifndef NN_CLI_TRAININGTUI_HPP
#define NN_CLI_TRAININGTUI_HPP

#include <functional>
#include <memory>

#include <sys/types.h>

namespace NN_CLI
{

  class TerminalUI;

  // Owns the "Loading samples" bar lifecycle during training and keeps it wired to a TerminalUI
  // across terminal resizes. Shared by the ANN and CNN runners: their cores differ, but the TUI
  // plumbing (GPU-count reservation, loading callback, resize repaint) is identical.
  class TrainingTui
  {
    public:
      // Bind to the training TUI and register the overlay that repaints the loading bar after a
      // resize. `onResize` (may be empty) is forwarded as the TUI resize callback, e.g. to reflow
      // the Config panel at the new width.
      void attach(std::shared_ptr<TerminalUI> tui, std::function<void()> onResize = {});

      // Reserve the per-GPU suffix width so the loading bar lines up with the epoch bar. The
      // prefetch loader runs ahead of the first training update, so the GPU count is resolved up
      // front (same logic the GPU core uses) rather than discovered dynamically.
      // deviceIsGpu / numGpusConfig come from the (library-specific) core config.
      void resolveBarGpus(bool deviceIsGpu, int numGpusConfig);

      // DataLoader loading callback: remembers progress, services a pending resize, repaints the bar.
      std::function<void(ulong, ulong, ulong, ulong)> loadingCallback();

      // Loading is over once the training stream starts consuming (call before core->train()).
      void markLoadingFinished()
      {
        this->loading_ = false;
      }

      // Force the loading bar to show the current batch as fully loaded.
      // Used after validation to catch up loading callbacks that were
      // suppressed while setLoadingEnabled(false) was active.
      void markCurrentLoadComplete();

    private:
      // Repaint the loading bar from the remembered state (no-op while not loading).
      void renderBar();

      std::shared_ptr<TerminalUI> tui_;
      int barGpus_ = 1;

      //-- Loading-bar state, re-rendered on resize --//
      ulong current_ = 0;
      ulong total_ = 0;
      ulong batchNum_ = 0;
      ulong totalBatches_ = 0;
      bool loading_ = false;
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGTUI_HPP
