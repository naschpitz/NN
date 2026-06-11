#ifndef NN_CLI_TERMINALUI_TRAININGWINDOW_HPP
#define NN_CLI_TERMINALUI_TRAININGWINDOW_HPP

#include "NN-CLI_TerminalUI_ModelInfoTable.hpp"
#include "NN-CLI_TerminalUI_Panel.hpp"
#include "NN-CLI_TerminalUI_ProgressBar.hpp"
#include "NN-CLI_TerminalUI_Table.hpp"
#include "NN-CLI_TerminalUI_Window.hpp"

#include <string>
#include <vector>

namespace NN_CLI
{

  // Specialized TerminalUI_Window for training mode.  Acts as the "View"
  // in the MVC pattern — it contains no training logic, only layout and
  // display of data provided by the Controller.
  //
  // Owns four child panels arranged in a responsive grid:
  //
  //   +-------------------------------------------+------------------+
  //   |         modelInfoPanel (top-left)         |                  |
  //   |         Model configuration / hyperparams |   timingPanel    |
  //   +-------------------------------------------+   (right column) |
  //   |         epochsPanel (bottom-left)         |                  |
  //   |         Epoch loss / validation table     |                  |
  //   +-------------------------------------------+------------------+
  //   |         progressPanel (bottom, full width)                   |
  //   |         Epoch + sample progress bars                         |
  //   +--------------------------------------------------------------+
  //
  // Each panel owns its internal widgets (tables, progress bars).
  // The public API allows the Controller to update panel data; the
  // window handles layout, drawing, and input routing.

  class TerminalUI_TrainingWindow : public TerminalUI_Window
  {
    public:
      //-- Types --//

      // Panel selection indices used by setActivePanel / getActivePanel.
      enum PanelIndex
      {
        MODEL_INFO = 0,
        EPOCHS = 1,
        TIMING = 2
      };

      //-- Ctors / Dtors --//

      TerminalUI_TrainingWindow();

      ~TerminalUI_TrainingWindow() override;

      //-- Layout --//

      // Reposition and resize the four child panels according to the
      // current window geometry.  Called automatically by resize() and
      // init().
      void layoutChildren() override;

      //-- Progress updates --//

      // Set single-segment progress data for the progress bar.
      void updateProgress(const std::string& label, float fraction);

      // Set multi-segment progress data (e.g. per-GPU fractions).
      void updateProgress(const std::string& label, const std::vector<float>& fractions);

      // Set the sub-line text rendered below the progress bar.
      void updateProgressSubLine(const std::string& text, int colorPair = 0);

      // Clear the sub-line text below the progress bar.
      void clearProgressSubLine();

      // Set a loading-phase progress bar with the given fraction.  Used to
      // display the "sample loading" phase before training begins.
      void setLoadingProgress(float fraction);

      // Clear the loading-phase progress bar.
      void clearLoadingProgress();

      //-- Epoch table --//

      // Replace the epoch table column definitions.
      void setEpochColumns(std::vector<TerminalUI_Table::Column> columns);

      // Append a single row of epoch data.
      void addEpochRow(const TerminalUI_Table::Row& row);

      // Append multiple rows of epoch data.
      void addEpochRows(const std::vector<TerminalUI_Table::Row>& rows);

      // Remove all epoch rows.
      void clearEpochRows();

      // Append a status / monitor message below the epoch table.
      void addEpochMessage(const std::string& message);

      // Remove all epoch messages.
      void clearEpochMessages();

      // Rebuild the epoch panel content from the current table rows and
      // messages, then apply it to the epochs panel.
      void refreshEpochContent();

      //-- Model info --//

      // Set the model info table title (centered header row).
      void setModelInfoTitle(const std::string& title);

      // Replace all model info entries.
      void setModelInfoEntries(const std::vector<TerminalUI_ModelInfoTable::Entry>& entries);

      // Append a single model info entry.
      void addModelInfoEntry(const std::string& key, const std::string& value);

      // Remove all model info entries.
      void clearModelInfoEntries();

      // Rebuild the model info panel content from the current entries and
      // apply it to the model info panel.
      void refreshModelInfoContent();

      //-- Timing --//

      // Store raw timing lines (will be padded on refresh).
      void setTimingLines(const std::vector<std::string>& lines);

      // Pad the stored timing lines to the timing panel width and apply
      // them to the panel.
      void refreshTimingContent();

      //-- Panel selection --//

      // Switch the active (highlighted) panel.  Valid values are the
      // PanelIndex enum constants.
      void setActivePanel(int panelIndex);

      int getActivePanel() const;

      //-- Panel access --//

      // Direct access to child panels (for scroll state, direct content
      // manipulation, etc.).
      TerminalUI_Panel* getProgressPanel() const;
      TerminalUI_Panel* getEpochsPanel() const;
      TerminalUI_Panel* getModelInfoPanel() const;
      TerminalUI_Panel* getTimingPanel() const;

      //-- Widget overrides --//

      void draw() override;
      bool handleEvent(int ch) override;

    protected:
      //-- Methods --//

      // Apply active / inactive color pairs to the panels.
      void updatePanelColors();

    private:
      //-- Members --//

      // Raw pointers to the child panels (owned via the base-class children
      // vector through unique_ptr).
      TerminalUI_Panel* progressPanelPtr = nullptr;
      TerminalUI_Panel* epochsPanelPtr = nullptr;
      TerminalUI_Panel* modelInfoPanelPtr = nullptr;
      TerminalUI_Panel* timingPanelPtr = nullptr;

      // Raw pointer to the progress bar widget (owned as a child of
      // progressPanel).
      TerminalUI_ProgressBar* progressBarPtr = nullptr;

      // Internal tables used to render formatted content before pushing
      // lines to the panels.  Not part of the widget hierarchy.
      TerminalUI_Table epochTable;
      TerminalUI_ModelInfoTable modelInfoTable;

      // Raw timing lines stored for re-padding on resize.
      std::vector<std::string> rawTimingLines;

      // Additional status / monitor messages displayed below the epoch
      // table.
      std::vector<std::string> epochMessages;

      // Currently highlighted panel index (PanelIndex values).
      int activePanel = 0;

      //-- Layout constants --//

      static constexpr int kProgressHeight = 5;
      static constexpr int kMinTimingWidth = 20;
      static constexpr int kMinLeftWidth = 68;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_TRAININGWINDOW_HPP
