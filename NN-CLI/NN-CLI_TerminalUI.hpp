#ifndef NN_CLI_TERMINALUI_HPP
#define NN_CLI_TERMINALUI_HPP

#include <atomic>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  class TerminalUI
  {
    public:
      struct EpochRecord {
          int epoch;
          float loss;
          bool hasValLoss;
          float valLoss;
          bool isBest;
          std::time_t completionTime; // when the epoch completed
      };

      TerminalUI();
      ~TerminalUI();

      bool init();
      void shutdown();

      bool isInitialized() const
      {
        return this->initialized_;
      }

      int rows() const
      {
        return this->rows_;
      }

      int cols() const
      {
        return this->cols_;
      }

      int timingWidth() const
      {
        return this->timingWidth_;
      }

      int leftWidth() const
      {
        return this->leftWidth_;
      }

      void setConfigLines(const std::vector<std::string>& lines);
      void refreshConfigPanel();

      WINDOW* progressWindow() const
      {
        return this->progressWin_;
      }

      WINDOW* loadingWindow() const
      {
        return this->loadingWin_;
      }

      void setTimingLines(const std::vector<std::string>& lines);
      void addEpochLine(const std::string& line);
      void pushEpochRecord(int epoch, float loss, bool hasValLoss, float valLoss, bool isBest,
                           std::time_t completionTime = 0);
      // Rebuild the epoch table lines from epochRecords_ using the current panel width.
      // Called on new epochs and on terminal resize so the table always fills the panel.
      void requestResize();
      bool handleResize();
      void redraw();
      void refresh();
      void pollInput();

      void setResizeCallback(std::function<void()> callback)
      {
        this->resizeCallback_ = std::move(callback);
      }

      // Invoked after the panels are redrawn (panel frames pushed to stdscr) so the caller can
      // repaint transient sub-window overlays — e.g. the loading bar, which lives in its own
      // window that layout() erases on resize and that nothing else redraws between callback ticks.
      void setOverlayCallback(std::function<void()> callback)
      {
        this->overlayCallback_ = std::move(callback);
      }

      std::recursive_mutex& mutex()
      {
        return this->mutex_;
      }

      // Read-only access to the structured epoch records (for merging history
      // before saving the final model).
      const std::vector<EpochRecord>& getEpochRecords() const
      {
        return this->epochRecords_;
      }

    private:
      // Scroll offset for one panel. `autoScroll` (Epochs only) keeps the view pinned to the
      // newest line until the user scrolls manually.
      struct ScrollState {
          int offset = 0;
          bool autoScroll = false;
      };

      void layout();
      void drawPanelFrame(int y, int h, const char* title, int titleColor = 2);
      void drawPanelFrame(int y, int h, int x, int w, const char* title, int titleColor);
      void drawAllPanels();

      // Composite the screen: redraw panels into stdscr, then layer the sub-windows on top.
      // runOverlay re-renders the loading-bar overlay (after layout() recreates the windows);
      // touchSub forces a full re-copy of the sub-windows when their content is otherwise unchanged.
      void present(bool runOverlay, bool touchSub);
      // Rebuild the epoch table lines from epochRecords_ using the current panel width.
      // Called on new epochs and on terminal resize so the table always fills the panel.
      void rebuildEpochLines();

      // Draw a vertical scrollbar in column `col` over `contentH` rows starting at `yTop`,
      // with the thumb positioned for `scroll` within [0, total - contentH]. No-op if it all fits.
      void drawScrollbar(int col, int yTop, int contentH, int scroll, int total);

      bool handleScrollInput(int ch);

      // Apply a scroll keypress to `s`; returns true if `ch` was a recognized scroll key.
      bool applyScroll(ScrollState& s, int ch, int contentH, int total);

      WINDOW* progressWin_ = nullptr;
      WINDOW* loadingWin_ = nullptr;
      WINDOW* timingWin_ = nullptr;

      int rows_ = 0;
      int cols_ = 0;
      bool initialized_ = false;

      int leftWidth_ = 0;
      int timingWidth_ = 0;

      int configY_ = 0;
      int configH_ = 0;
      int trainingY_ = 0;
      int trainingH_ = 0;
      int epochsY_ = 0;
      int epochsH_ = 0;
      int helpY_ = 0;

      int activePanel_ = 0; // 0=Config, 1=Epochs, 2=Timing
      ScrollState config_;
      ScrollState epochs_{0, true};
      ScrollState timing_;

      std::recursive_mutex mutex_;
      std::atomic<uint> resizeRequested_{0};
      std::function<void()> resizeCallback_;
      std::function<void()> overlayCallback_;

      std::vector<std::string> configLines_;
      std::vector<std::string> timingLines_;
      std::vector<std::string> epochLines_;
      std::vector<EpochRecord> epochRecords_; // structured epoch data (for future use)
      std::vector<std::string> epochMessages_; // monitor/status messages preserved across table rebuilds

      bool epochLinesDirty_{true};
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_HPP
