#ifndef NN_CLI_PROGRESSBAR_HPP
#define NN_CLI_PROGRESSBAR_HPP

#include <chrono>
#include <mutex>
#include <ostream>
#include <vector>

#include <sys/types.h>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Common progress info struct used by both ANN and CNN training callbacks
  struct ProgressInfo {
      ulong currentEpoch;
      ulong totalEpochs;
      ulong currentSample;
      ulong totalSamples;
      float epochLoss;
      float sampleLoss;
      int gpuIndex;
      int totalGPUs;
  };

  class ProgressBar
  {
    public:
      ProgressBar(ulong progressReports = 1000, int barWidth = 50);

      // Update and display progress (call from training callback).
      // If `win` is non-null, renders into the ncurses window.
      void update(const ProgressInfo& progress, WINDOW* win = nullptr);

      // When true, the epoch-complete line is left open (no trailing newline) so the
      // caller can append to it (e.g. " - Validation Loss: ...") and commit it itself.
      // When false, the bar commits the epoch line with its own newline.
      // Only meaningful for non-ncurses (std::cout) rendering.
      void setHoldEpochLine(bool hold)
      {
        this->holdEpochLine = hold;
      }

      // Simple loading progress bar (static, self-contained).
      static void printLoadingProgress(const std::string& label, size_t current, size_t total,
                                       ulong progressReports = 1000, int barWidth = 40);

      // Write a status message on line 2 of the progress window (e.g. validation progress).
      static void writeStatus(WINDOW* win, const std::string& msg);

      // Clear the status line (line 2) of the progress window.
      static void clearStatus(WINDOW* win);

      // Replace the epoch progress bar (line 0) with a validation bar.
      // Renders "Validating [████░░░░] XX.X%" using the full window width.
      static void renderValidationBar(WINDOW* win, float pct);

      // Render a determinate loading progress bar into the ncurses progress window.
      // batchNum and totalBatches are displayed in the label (e.g. "Loading samples (1/3): [████░░] 32/64").
      // numGpus must match the epoch bar's GPU count so both bars reserve the same right-side
      // space and line up vertically (the loader runs ahead of training, so it must be told).
      static void renderLoadingBar(WINDOW* win, ulong current, ulong total, ulong batchNum = 1, ulong totalBatches = 1,
                                   int numGpus = 1);

    private:
      //-- Configuration --//
      ulong progressReports;
      int barWidth;

      //-- Per-epoch throughput timer (images/second + ETA) --//
      ulong timerEpoch = 0;
      std::chrono::steady_clock::time_point epochStartTime;

      //-- Running loss accumulator (smoothed per-epoch average) --//
      double runningLossSum = 0.0;
      ulong runningLossCount = 0;

      //-- Multi-GPU progress tracking --//
      std::mutex mutex;
      std::vector<float> gpuProgress;
      int totalGPUs = 0;
      ulong currentEpoch = 0;

      //-- Epoch-line ownership: when held, caller appends & commits --//
      bool holdEpochLine = false;

      //-- Internal methods --//
      void resetGpuState(int numGPUs, ulong epoch);
      void updateGpuProgress(int gpuIndex, float percent);
      std::vector<float> getGpuProgress();

      //-- Rendering --//
      void renderSingleBar(std::ostream& out, float percent);
      void renderMultiGpuBar(std::ostream& out, const std::vector<float>& gpuProg, int numGPUs);

      // ncurses rendering
      void renderNcursesBar(WINDOW* win, float percent, int barWidth);
      void renderNcursesMultiBar(WINDOW* win, const std::vector<float>& gpuProg, int numGPUs, int barWidth);
  };

} // namespace NN_CLI

#endif // NN_CLI_PROGRESSBAR_HPP
