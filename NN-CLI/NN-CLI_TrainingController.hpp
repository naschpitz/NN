#ifndef NN_CLI_TRAININGCONTROLLER_HPP
#define NN_CLI_TRAININGCONTROLLER_HPP

#include "NN-CLI_DataLoader.hpp"

#include "NN-CLI_TrainingProgressTracker.hpp"

#include <functional>
#include <memory>

#include <sys/types.h>

namespace NN_CLI
{

  class TerminalUI;

  // Owns the "Samples" bar lifecycle during training and keeps it wired to a TerminalUI
  // across terminal resizes. Shared by the  and CNN runners: their cores differ, but the TUI
  // plumbing (GPU-count reservation, loading callback, resize repaint) is identical.
  class TrainingController
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
      std::function<void(ulong, ulong, ulong, ulong, SampleLoadType)> loadingCallback();

      // Loading is over once the training stream starts consuming (call before core->train()).
      void markLoadingFinished()
      {
        this->loading = false;
      }

      // Initialize the training progress tracker with the given parameters.
      void initTracker(ulong progressReports, ulong windowSize, int barWidth);

      // Forward progress info to the tracker for ncurses rendering.
      void updateProgress(const ProgressInfo& progress);

    private:
      // Repaint the loading bar from the remembered state (no-op while not loading).
      void renderBar();

      std::shared_ptr<TerminalUI> tui;
      int barGpus = 1;

      //-- Training progress tracker --//
      std::unique_ptr<TrainingProgressTracker> progressTracker;

      //-- Loading-bar state, re-rendered on resize --//
      ulong current = 0;
      ulong total = 0;
      ulong batchNum = 0;
      ulong totalBatches = 0;
      bool loading = false;
  };

} // namespace NN_CLI

#endif // NN_CLI_TRAININGCONTROLLER_HPP
