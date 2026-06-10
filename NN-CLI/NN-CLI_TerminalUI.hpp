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
        return this->initialized;
      }

      //-- Accessors --//

      int getRows() const
      {
        return this->rows;
      }

      int getCols() const
      {
        return this->cols;
      }

      int getTimingWidth() const
      {
        return this->timingWidth;
      }

      int getLeftWidth() const
      {
        return this->leftWidth;
      }

      // Return the content width for the Configuration panel, dynamically accounting
      // for whether a scrollbar is needed — the same logic used by renderConfigContent().
      int configContentWidth() const;

      WINDOW* progressWindow() const
      {
        return this->progressWin;
      }

      WINDOW* loadingWindow() const
      {
        return this->loadingWin;
      }

      std::recursive_mutex& getMutex()
      {
        return this->mutex;
      }

      // Read-only access to the structured epoch records (for merging history
      // before saving the final model).
      const std::vector<EpochRecord>& getEpochRecords() const
      {
        return this->epochRecords;
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
        this->resizeCallback = std::move(callback);
      }

      // Invoked after the panels are redrawn (panel frames pushed to stdscr) so the caller can
      // repaint transient sub-window overlays — e.g. the loading bar, which lives in its own
      // window that layout() erases on resize and that nothing else redraws between callback ticks.
      void setOverlayCallback(std::function<void()> callback)
      {
        this->overlayCallback = std::move(callback);
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

      WINDOW* progressWin = nullptr;
      WINDOW* loadingWin = nullptr;

      //-- Terminal state --//

      int rows = 0;
      int cols = 0;
      bool initialized = false;

      int leftWidth = 0;
      int timingWidth = 0;

      //-- Panels --//

      TerminalUI_Panel trainingPanel;
      TerminalUI_Panel epochsPanel;
      TerminalUI_Panel configPanel;
      TerminalUI_Panel timingPanel;

      //-- Tables --//

      TerminalUI_Table epochsTable;

      //-- Panel selection --//

      int activePanel = 0; // 0=Config, 1=Epochs, 2=Timing

      //-- Synchronization --//

      mutable std::recursive_mutex mutex;
      std::atomic<uint> resizeRequested{0};
      std::function<void()> resizeCallback;
      std::function<void()> overlayCallback;

      //-- Data --//

      std::vector<SummaryTable::Section> configSections;
      std::vector<EpochRecord> epochRecords;
      std::vector<std::string> epochMessages; // monitor/status messages preserved across table rebuilds

      //-- Dirty flags --//

      bool epochLinesDirty{true};
      bool configLinesDirty{true};
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_HPP
