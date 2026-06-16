#ifndef NN_CLI_TERMINALUI_PREDICTWINDOW_HPP
#define NN_CLI_TERMINALUI_PREDICTWINDOW_HPP

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

  // Specialized TerminalUI_Window for predict mode.  Acts as the "View"
  // in the MVC pattern — it contains no prediction logic, only layout and
  // display of data provided by the Controller.
  //
  // Owns five child panels arranged in a responsive grid:
  //
  //   +-------------------------------------+------------------------+
  //   |         modelInfoPanel (top-left)   |     timingPanel       |
  //   |         Model configuration /       |     Prediction timing  |
  //   |         architecture info           |                        |
  //   +-------------------------------------+------------------------+
  //   |         epochHistoryPanel (bot-left)|     resultsPanel       |
  //   |         Training metadata           |     {Index, Class,     |
  //   |         (raw lines)                 |      Confidence}       |
  //   +-------------------------------------+------------------------+
  //   |              progressPanel (bottom, full width)              |
  //   |              Prediction progress bar                         |
  //   +--------------------------------------------------------------+
  //
  // Each panel owns its internal widgets (tables, progress bars).
  // The public API allows the Controller to update panel data; the
  // window handles layout, drawing, input routing, and dismiss handling.

  class TerminalUI_PredictWindow : public TerminalUI_Window
  {
    public:
      //-- Types --//

      // Panel selection indices used by setActivePanel / getActivePanel.
      // Note: PROGRESS (4) is part of the tab cycle for visual consistency
      // but is always rendered with an inactive color pair.
      enum PanelIndex { MODEL_INFO = 0, TIMING = 1, EPOCH_HISTORY = 2, RESULTS = 3, PROGRESS = 4, PANEL_COUNT = 5 };

      //-- Ctors / Dtors --//

      TerminalUI_PredictWindow();

      ~TerminalUI_PredictWindow() override;

      //-- Layout --//

      // Reposition and resize the five child panels according to the
      // current window geometry.  Called automatically by resize() and
      // init().
      void layoutChildren() override;

      //-- Progress updates --//

      // Set single-segment progress data for the progress bar.
      void updateProgress(const std::string& label, float fraction);

      // Set multi-segment progress data (e.g. per-GPU fractions).
      void updateProgress(const std::string& label, const std::vector<float>& fractions);

      // Set the sub-line text rendered below the progress bar.
      void updateProgressSubLine(const std::string& line);

      // Clear the sub-line text below the progress bar.
      void clearProgressSubLine();

      // Set the dedicated "sample loading" progress bar (shown before
      // prediction begins).  Makes the bar visible on first use.
      void setLoadingProgress(const std::string& label, float fraction);

      // Hide the loading-phase progress bar.
      void clearLoadingProgress();

      //-- Model Info panel --//

      // Replace all model info rows directly (preserving section
      // separators encoded as empty-key rows).
      void setModelInfoRows(const std::vector<SummaryRow>& rows);

      // Rebuild the model info panel content from the current rows
      // and apply it to the model info panel.
      void refreshModelInfoContent();

      //-- Timing panel --//

      // Store raw timing lines (will be padded on refresh).
      void setTimingLines(const std::vector<std::string>& lines);

      // Pad the stored timing lines to the timing panel width and apply
      // them to the panel.
      void refreshTimingContent();

      //-- Epoch History panel --//

      // Store raw epoch-history lines (will be padded on refresh).
      void setEpochHistoryLines(const std::vector<std::string>& lines);

      // Pad the stored epoch-history lines to the panel width and apply
      // them to the panel.
      void refreshEpochHistoryContent();

      //-- Results panel --//

      // Set the Results table column definitions.
      void setResultsColumns(const std::vector<std::string>& columns);

      // Append a single row of prediction results.
      void addResultRow(const std::vector<std::string>& row);

      // Append multiple rows of prediction results.
      void addResultRows(const std::vector<std::vector<std::string>>& rows);

      // Remove all result rows.
      void clearResultRows();

      // Rebuild the Results panel content from the current table data
      // and apply it to the panel.
      void refreshResultsContent();

      //-- Panel selection --//

      // Switch the active (highlighted) panel.  Valid values are the
      // PanelIndex enum constants.
      void setActivePanel(int panelIndex);

      int getActivePanel() const;

      //-- Panel access --//

      // Direct access to child panels (for scroll state, direct content
      // manipulation, etc.).
      TerminalUI_Panel* getProgressPanel() const;
      TerminalUI_Panel* getModelInfoPanel() const;
      TerminalUI_Panel* getTimingPanel() const;
      TerminalUI_Panel* getEpochHistoryPanel() const;
      TerminalUI_Panel* getResultsPanel() const;

      //-- Event routing --//

      // Handle a Tab keypress to cycle the active panel
      // (wraps 0→1→2→3→4→0).  Returns true if ch was Tab and was consumed.
      bool cycleActivePanel(int ch);

      // Route a scroll keypress to the active panel's applyScrollInput().
      // Returns true if ch was a recognized scroll key.
      bool scrollActivePanel(int ch);

      //-- Dismiss handling --//

      // Spin-loop waiting for the user to press 'q' or ESC.
      // Does NOT hold the window mutex.
      void waitForDismiss();

      //===================================================================================================================//
      // Return true if the user has requested an abort (pressed 'q', 'Q', or ESC).
      bool abortRequested() const
      {
        return this->dismissed_.load();
      }

      //-- Widget overrides --//

      bool handleEvent(int ch) override;

    protected:
      //-- Hooks --//

      // Apply active / inactive panel color pairs before rendering.
      void preRender() override;

      //-- Methods --//

      // Apply active / inactive color pairs to the panels.
      void updatePanelColors();

      // Grant both progress bars the widest label and per-segment suffix
      // either needs so their brackets stay vertically aligned (comparable
      // progress) while the bars take every remaining column.
      void syncProgressBarLayout();

    private:
      //-- Members --//

      // Raw pointers to the child panels (owned via the base-class children
      // vector through unique_ptr).
      TerminalUI_Panel* progressPanelPtr = nullptr;
      TerminalUI_Panel* modelInfoPanelPtr = nullptr;
      TerminalUI_Panel* timingPanelPtr = nullptr;
      TerminalUI_Panel* epochHistoryPanelPtr = nullptr;
      TerminalUI_Panel* resultsPanelPtr = nullptr;

      // Raw pointers to the progress bar widgets (owned as children of
      // progressPanel).  loadingBarPtr is the top "Samples" bar;
      // progressBarPtr is the prediction bar below it.
      TerminalUI_ProgressBar* loadingBarPtr = nullptr;
      TerminalUI_ProgressBar* progressBarPtr = nullptr;

      // Internal results table used to render formatted content before
      // pushing lines to the results panel.  Not part of the widget
      // hierarchy.
      TerminalUI_Table resultsTable;

      // Summary rows used to render the model info panel via SummaryTable.
      std::vector<SummaryRow> modelInfoRows;

      // Raw timing lines stored for re-padding on resize.
      std::vector<std::string> rawTimingLines;

      // Raw epoch-history lines stored for re-padding on resize.
      std::vector<std::string> rawEpochHistoryLines;

      // Currently highlighted panel index (PanelIndex values).
      int activePanel = 0;

      // Set to true when the user presses a dismiss key.
      std::atomic<bool> dismissed_{false};

      //-- Layout constants --//

      static constexpr int kProgressHeight = 3;
      static constexpr int kMinTimingWidth = 25;
      static constexpr int kMinLeftWidth = 40;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PREDICTWINDOW_HPP
