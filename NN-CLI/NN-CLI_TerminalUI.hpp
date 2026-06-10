#ifndef NN_CLI_TERMINALUI_HPP
#define NN_CLI_TERMINALUI_HPP

#include <atomic>
#include <ctime>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_TerminalUI_Panel.hpp"
#include "NN-CLI_TerminalUI_Table.hpp"

struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  class TerminalUI
  {
    public:
      //-- Types --//

      struct EpochRecord {
          int epoch;
          float loss;
          bool hasValLoss;
          float valLoss;
          bool isBest;
          std::time_t completionTime; // when the epoch completed
      };

      //-- Ctors / Dtors --//

      TerminalUI();
      ~TerminalUI();

      //-- Lifecycle --//

      bool init();
      void shutdown();

      bool isInitialized() const
      {
        return this->initialized_;
      }

      //-- Accessors --//

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

      // Return the content width for the Configuration panel, dynamically accounting
      // for whether a scrollbar is needed — the same logic used by renderConfigContent().
      int configContentWidth() const;

      WINDOW* progressWindow() const
      {
        return this->progressWin_;
      }

      WINDOW* loadingWindow() const
      {
        return this->loadingWin_;
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

      //-- Content --//

      void setConfigLines(const std::vector<std::string>& lines);

      void setConfigSections(const std::vector<SummaryTable::Section>& sections);
      void refreshConfigPanel();

      void setTimingLines(const std::vector<std::string>& lines);
      void addEpochLine(const std::string& line);
      void pushEpochRecord(int epoch, float loss, bool hasValLoss, float valLoss, bool isBest,
                           std::time_t completionTime = 0);

      //-- Input / Refresh --//

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

    private:
      //-- Methods --//

      void layout();
      void drawAllPanels();

      // Composite the screen: redraw panels into stdscr, then layer the sub-windows on top.
      // runOverlay re-renders the loading-bar overlay (after layout() recreates the windows);
      // touchSub forces a full re-copy of the sub-windows when their content is otherwise unchanged.
      void present(bool runOverlay, bool touchSub);

      bool handleScrollInput(int ch);

      // Rebuild panel content from structured data using the table/panel classes.
      // Called from present() when the corresponding dirty flag is set.
      void renderEpochContent();
      void renderConfigContent();

      //-- Sub-windows (overlays) --//

      WINDOW* progressWin_ = nullptr;
      WINDOW* loadingWin_ = nullptr;

      //-- Terminal state --//

      int rows_ = 0;
      int cols_ = 0;
      bool initialized_ = false;

      int leftWidth_ = 0;
      int timingWidth_ = 0;

      //-- Panels --//

      TerminalUI_Panel trainingPanel_;
      TerminalUI_Panel epochsPanel_;
      TerminalUI_Panel configPanel_;
      TerminalUI_Panel timingPanel_;

      //-- Tables --//

      TerminalUI_Table epochsTable_;

      //-- Panel selection --//

      int activePanel_ = 0; // 0=Config, 1=Epochs, 2=Timing

      //-- Synchronization --//

      mutable std::recursive_mutex mutex_;
      std::atomic<uint> resizeRequested_{0};
      std::function<void()> resizeCallback_;
      std::function<void()> overlayCallback_;

      //-- Data --//

      std::vector<SummaryTable::Section> configSections_;
      std::vector<EpochRecord> epochRecords_;
      std::vector<std::string> epochMessages_; // monitor/status messages preserved across table rebuilds

      //-- Dirty flags --//

      bool epochLinesDirty_{true};
      bool configLinesDirty_{true};
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_HPP
