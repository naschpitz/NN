#ifndef NN_CLI_TERMINALUI_RESULTSWINDOW_HPP
#define NN_CLI_TERMINALUI_RESULTSWINDOW_HPP

#include "NN-CLI_SummaryTable.hpp"
#include "NN-CLI_TerminalUI_Panel.hpp"
#include "NN-CLI_TerminalUI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI_Table.hpp"
#include "NN-CLI_TerminalUI_Window.hpp"

#include <atomic>
#include <string>
#include <utility>
#include <vector>

namespace NN_CLI
{

  // Shared base for the evaluation-style TUI windows (predict, test, and later
  // calibrate).  All three present the same five-panel layout:
  //
  //   +-------------------------------------+------------------------+
  //   |         modelInfoPanel (top-left)   |     timingPanel        |
  //   |         Model configuration /       |     Run timing         |
  //   |         architecture info           |                        |
  //   +-------------------------------------+------------------------+
  //   |     epochHistoryPanel (bottom-left) |     resultsPanel       |
  //   |     Training metadata (raw lines)   |     results table      |
  //   +-------------------------------------+------------------------+
  //   |              progressPanel (bottom, full width)              |
  //   |              progress bar                                    |
  //   +--------------------------------------------------------------+
  //
  // Subclasses only customize the window title/color and the results-table
  // columns via the constructor — every layout, progress, scroll, and dismiss
  // behavior is shared here so the per-mode window classes stay trivial and
  // nothing is duplicated.

  class TerminalUI_ResultsWindow : public TerminalUI_Window
  {
    public:
      //-- Types --//

      // Panel selection indices used by setActivePanel / getActivePanel.
      // Note: PROGRESS (4) is part of the tab cycle for visual consistency
      // but is always rendered with an inactive color pair.
      enum PanelIndex { MODEL_INFO = 0, TIMING = 1, EPOCH_HISTORY = 2, RESULTS = 3, PROGRESS = 4, PANEL_COUNT = 5 };

      //-- Ctors / Dtors --//

      TerminalUI_ResultsWindow(const std::string& title, int titleColor,
                               const std::vector<TerminalUI_Table::Column>& defaultResultColumns,
                               std::vector<int> resultColumnWidthHints);

      ~TerminalUI_ResultsWindow() override;

      //-- Layout --//

      void layoutChildren() override;

      //-- Progress updates --//

      void updateProgress(const std::string& label, float fraction);

      void updateProgress(const std::string& label, const std::vector<float>& fractions);

      void updateProgressSubLine(const std::string& line);

      void clearProgressSubLine();

      void setLoadingProgress(const std::string& label, float fraction);

      void clearLoadingProgress();

      //-- Model Info panel --//

      void setModelInfoRows(const std::vector<SummaryRow>& rows);

      void refreshModelInfoContent();

      //-- Timing panel --//

      void setTimingLines(const std::vector<std::string>& lines);

      void refreshTimingContent();

      //-- Epoch History panel --//

      void setEpochHistoryLines(const std::vector<std::string>& lines);

      void refreshEpochHistoryContent();

      //-- Results panel --//

      void setResultsColumns(const std::vector<std::string>& columns);

      void addResultRow(const std::vector<std::string>& row);

      void addResultRows(const std::vector<std::vector<std::string>>& rows);

      void clearResultRows();

      void refreshResultsContent();

      //-- Panel selection --//

      void setActivePanel(int panelIndex);

      int getActivePanel() const;

      //-- Panel access --//

      TerminalUI_Panel* getProgressPanel() const;
      TerminalUI_Panel* getModelInfoPanel() const;
      TerminalUI_Panel* getTimingPanel() const;
      TerminalUI_Panel* getEpochHistoryPanel() const;
      TerminalUI_Panel* getResultsPanel() const;

      //-- Event routing --//

      bool cycleActivePanel(int ch);

      bool scrollActivePanel(int ch);

      //-- Dismiss handling --//

      void waitForDismiss();

      bool abortRequested() const
      {
        return this->dismissed.load();
      }

      //-- Widget overrides --//

      bool handleEvent(int ch) override;

    protected:
      //-- Hooks --//

      void preRender() override;

      //-- Methods --//

      void updatePanelColors();

      void syncProgressBarLayout();

    private:
      //-- Members --//

      TerminalUI_Panel* progressPanelPtr = nullptr;
      TerminalUI_Panel* modelInfoPanelPtr = nullptr;
      TerminalUI_Panel* timingPanelPtr = nullptr;
      TerminalUI_Panel* epochHistoryPanelPtr = nullptr;
      TerminalUI_Panel* resultsPanelPtr = nullptr;

      TerminalUI_ProgressBar* loadingBarPtr = nullptr;
      TerminalUI_ProgressBar* progressBarPtr = nullptr;

      TerminalUI_Table resultsTable;

      std::vector<SummaryRow> modelInfoRows;

      std::vector<std::string> rawTimingLines;

      std::vector<std::string> rawEpochHistoryLines;

      // Per-column width hints used by setResultsColumns (falls back to 10 for
      // any column index beyond the vector's size).
      std::vector<int> resultColumnWidthHints;

      int activePanel = 0;

      std::atomic<bool> dismissed{false};

      //-- Layout constants --//

      static constexpr int kProgressHeight = 3;
      static constexpr int kMinTimingWidth = 25;
      static constexpr int kMinLeftWidth = 40;
  };

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_TERMINALUI_RESULTSWINDOW_HPP
