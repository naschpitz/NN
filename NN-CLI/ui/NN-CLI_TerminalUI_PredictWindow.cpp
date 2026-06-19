#include "NN-CLI_TerminalUI_PredictWindow.hpp"

#include <QThread>

#include <algorithm>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_PredictWindow::TerminalUI_PredictWindow()
  {
    //-- Create and register child panels --//
    // Order: [0]=progress, [1]=modelInfo, [2]=timing, [3]=epochHistory, [4]=results

    auto progressPanel = std::make_unique<TerminalUI_Panel>("Progress", 2);
    auto modelInfoPanel = std::make_unique<TerminalUI_Panel>("Model Info", 2);
    auto timingPanel = std::make_unique<TerminalUI_Panel>("Timing", 2);
    auto epochHistoryPanel = std::make_unique<TerminalUI_Panel>("Epoch History", 2);
    auto resultsPanel = std::make_unique<TerminalUI_Panel>("Results", 2);

    this->progressPanelPtr = progressPanel.get();
    this->modelInfoPanelPtr = modelInfoPanel.get();
    this->timingPanelPtr = timingPanel.get();
    this->epochHistoryPanelPtr = epochHistoryPanel.get();
    this->resultsPanelPtr = resultsPanel.get();

    this->addChild(std::move(progressPanel));
    this->addChild(std::move(modelInfoPanel));
    this->addChild(std::move(timingPanel));
    this->addChild(std::move(epochHistoryPanel));
    this->addChild(std::move(resultsPanel));

    //-- Configure panels --//

    this->modelInfoPanelPtr->setAutoScroll(true);
    this->timingPanelPtr->setAutoScroll(true);
    this->epochHistoryPanelPtr->setAutoScroll(true);
    this->resultsPanelPtr->setAutoScroll(true);

    //-- Create and attach the two stacked progress bars inside the progress --//
    //-- panel: the "Samples" loading bar on top, the prediction bar below.  --//

    auto loadingBar = std::make_unique<TerminalUI_ProgressBar>();
    this->loadingBarPtr = loadingBar.get();
    this->loadingBarPtr->setVisible(false); // hidden until the first sample load
    this->progressPanelPtr->addChild(std::move(loadingBar));

    auto progressBar = std::make_unique<TerminalUI_ProgressBar>();
    this->progressBarPtr = progressBar.get();
    this->progressPanelPtr->addChild(std::move(progressBar));

    //-- Configure the results table with default columns --//

    this->resultsTable.setColumns({
      {"Index", 6, TerminalUI_Table::Align::RIGHT},
      {"Predicted Class", 16, TerminalUI_Table::Align::LEFT},
      {"Confidence", 11, TerminalUI_Table::Align::RIGHT},
    });

    this->setShortcutBar("Tab: panels ↑↓: scroll PgUp/PgDn: page Home/End: top/bottom q/ESC: quit");
    this->setTitleBar("---- PREDICT ----", 1);
  }

  //===================================================================================================================//

  TerminalUI_PredictWindow::~TerminalUI_PredictWindow() {}

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::layoutChildren()
  {
    int W = this->width;
    int H = this->height - this->shortcutBarHeight() - this->titleBarHeight();
    int titleH = this->titleBarHeight();

    if (W <= 0 || H <= 0)
      return;

    //-- Reserve the progress panel at the bottom --//

    int progressH = std::min(kProgressHeight, H);
    int remainingH = std::max(0, H - progressH);

    //-- Horizontal split: right column (Timing + Results) if screen is wide enough --//

    int rightW = 0;
    int leftW = W;

    if (W >= kMinLeftWidth + kMinTimingWidth) {
      int idealRightW = W * 40 / 100;
      rightW = std::max(kMinTimingWidth, std::min(W - kMinLeftWidth, idealRightW));
      leftW = W - rightW;
    }

    //-- Vertical split of each column: top row ~45%, bottom row ~55% --//

    int topRowH = std::min(remainingH, std::max(3, remainingH * 45 / 100));
    int bottomRowH = std::max(0, remainingH - topRowH);

    //-- Position panels --//
    //   modelInfoPanel   (children[1]) — top-left:     (0, 0)            size: (leftW, topRowH)
    //   timingPanel      (children[2]) — top-right:    (leftW, 0)        size: (rightW, topRowH)
    //   epochHistoryPanel(children[3]) — bottom-left:  (0, topRowH)      size: (leftW, bottomRowH)
    //   resultsPanel     (children[4]) — bottom-right: (leftW, topRowH)  size: (rightW, bottomRowH)
    //   progressPanel    (children[0]) — bottom:       (0, H - progressH) size: (W, progressH)

    if (this->childCount() < 5)
      return;

    this->children[0]->resize(W, progressH, 0, H - progressH + titleH);
    this->children[1]->resize(leftW, topRowH, 0, titleH);
    this->children[2]->resize(rightW, topRowH, leftW, titleH);

    if (bottomRowH > 0) {
      this->children[3]->resize(leftW, bottomRowH, 0, topRowH + titleH);
      this->children[4]->resize(rightW, bottomRowH, leftW, topRowH + titleH);
    } else {
      this->children[3]->resize(0, 0, 0, titleH);
      this->children[4]->resize(0, 0, 0, titleH);
    }

    //-- Stack the two progress bars inside the progress panel --//

    int contentX = 2; // progress panel x (0) + border/pad
    int contentY = (H - progressH) + titleH + 1; // progress panel y + title bar offset + top border
    int contentW = std::max(1, W - 4);
    int contentH = std::max(0, progressH - 2);

    if (this->loadingBarPtr && this->progressBarPtr) {
      int loadingH = (contentH > 1) ? 1 : 0;
      int trainingH = std::max(0, contentH - loadingH);

      this->loadingBarPtr->resize(contentW, loadingH, contentX, contentY);
      this->progressBarPtr->resize(contentW, trainingH, contentX, contentY + loadingH);
    }

    // Rebuild panel content so that tables pick up the new column widths
    // after a terminal resize.
    this->refreshModelInfoContent();
    this->refreshTimingContent();
    this->refreshEpochHistoryContent();
    this->refreshResultsContent();
  }

  //===================================================================================================================//
  //-- Progress updates --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::updateProgress(const std::string& label, float fraction)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setBarData(label, fraction);

    this->syncProgressBarLayout();
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::updateProgress(const std::string& label, const std::vector<float>& fractions)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setBarData(label, fractions);

    this->syncProgressBarLayout();
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::updateProgressSubLine(const std::string& line)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setSubLineText(line, 0);
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::clearProgressSubLine()
  {
    if (this->progressBarPtr)
      this->progressBarPtr->clearSubLineText();
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::setLoadingProgress(const std::string& label, float fraction)
  {
    if (this->loadingBarPtr) {
      this->loadingBarPtr->setVisible(true);
      this->loadingBarPtr->setBarData(label, fraction);
    }

    this->syncProgressBarLayout();
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::clearLoadingProgress()
  {
    if (this->loadingBarPtr)
      this->loadingBarPtr->setVisible(false);
  }

  //===================================================================================================================//
  //-- Model Info --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::setModelInfoRows(const std::vector<SummaryRow>& rows)
  {
    this->modelInfoRows = rows;
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::refreshModelInfoContent()
  {
    if (!this->modelInfoPanelPtr)
      return;

    int tableWidth = std::max(30, this->modelInfoPanelPtr->contentWidth());

    SummaryTable::Section configSection;
    configSection.title = "Model Configuration";
    configSection.rows = this->modelInfoRows;

    std::vector<SummaryTable::Section> sections;
    sections.push_back(std::move(configSection));

    auto lines = SummaryTable::collectSections(sections, static_cast<ulong>(tableWidth));
    this->modelInfoPanelPtr->setLines(lines);
    this->modelInfoPanelPtr->setScrollOffset(0);
  }

  //===================================================================================================================//
  //-- Timing --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::setTimingLines(const std::vector<std::string>& lines)
  {
    this->rawTimingLines = lines;
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::refreshTimingContent()
  {
    if (!this->timingPanelPtr)
      return;

    int contentW = this->timingPanelPtr->contentWidth();
    auto lines = this->rawTimingLines;

    for (auto& line : lines) {
      int lineLen = static_cast<int>(line.size());

      if (lineLen < contentW)
        line.append(static_cast<std::string::size_type>(contentW - lineLen), ' ');
    }

    this->timingPanelPtr->setLines(lines);
  }

  //===================================================================================================================//
  //-- Epoch History --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::setEpochHistoryLines(const std::vector<std::string>& lines)
  {
    this->rawEpochHistoryLines = lines;
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::refreshEpochHistoryContent()
  {
    if (!this->epochHistoryPanelPtr)
      return;

    int contentW = this->epochHistoryPanelPtr->contentWidth();
    auto lines = this->rawEpochHistoryLines;

    for (auto& line : lines) {
      int lineLen = static_cast<int>(line.size());

      if (lineLen < contentW)
        line.append(static_cast<std::string::size_type>(contentW - lineLen), ' ');
    }

    this->epochHistoryPanelPtr->setLines(lines);
  }

  //===================================================================================================================//
  //-- Results --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::setResultsColumns(const std::vector<std::string>& columns)
  {
    if (columns.size() >= 1) {
      std::vector<TerminalUI_Table::Column> tableColumns;
      tableColumns.reserve(columns.size());

      for (ulong i = 0; i < columns.size(); i++) {
        int widthHint = 10;

        if (i == 0)
          widthHint = 6; // Index
        else if (i == 1)
          widthHint = 16; // Predicted Class
        else if (i == 2)
          widthHint = 11; // Confidence

        tableColumns.push_back({columns[i], widthHint, TerminalUI_Table::Align::LEFT});
      }

      this->resultsTable.setColumns(tableColumns);
    }
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::addResultRow(const std::vector<std::string>& row)
  {
    this->resultsTable.addRow(row);
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::addResultRows(const std::vector<std::vector<std::string>>& rows)
  {
    this->resultsTable.addRows(rows);
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::clearResultRows()
  {
    this->resultsTable.clearRows();
  }

  //===================================================================================================================//

  void TerminalUI_PredictWindow::refreshResultsContent()
  {
    if (!this->resultsPanelPtr)
      return;

    int tableWidth = std::max(20, this->resultsPanelPtr->contentWidth());
    this->resultsTable.setMaxWidth(tableWidth);

    auto lines = this->resultsTable.render();
    this->resultsPanelPtr->setLines(lines);
  }

  //===================================================================================================================//
  //-- Panel selection --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::setActivePanel(int panelIndex)
  {
    this->activePanel = panelIndex;
  }

  //===================================================================================================================//

  int TerminalUI_PredictWindow::getActivePanel() const
  {
    return this->activePanel;
  }

  //===================================================================================================================//
  //-- Panel access --//
  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_PredictWindow::getProgressPanel() const
  {
    return this->progressPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_PredictWindow::getModelInfoPanel() const
  {
    return this->modelInfoPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_PredictWindow::getTimingPanel() const
  {
    return this->timingPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_PredictWindow::getEpochHistoryPanel() const
  {
    return this->epochHistoryPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_PredictWindow::getResultsPanel() const
  {
    return this->resultsPanelPtr;
  }

  //===================================================================================================================//
  //-- Dismiss handling --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::waitForDismiss()
  {
    while (!this->dismissed.load())
      QThread::msleep(50);
  }

  //===================================================================================================================//
  //-- Hooks --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::preRender()
  {
    this->updatePanelColors();
  }

  //===================================================================================================================//
  //-- Event routing --//
  //===================================================================================================================//

  bool TerminalUI_PredictWindow::cycleActivePanel(int ch)
  {
    if (ch != '\t')
      return false;

    this->activePanel = (this->activePanel + 1) % PANEL_COUNT;
    return true;
  }

  //===================================================================================================================//

  bool TerminalUI_PredictWindow::scrollActivePanel(int ch)
  {
    TerminalUI_Panel* scrollablePanels[] = {
      this->modelInfoPanelPtr,
      this->timingPanelPtr,
      this->epochHistoryPanelPtr,
      this->resultsPanelPtr,
    };

    // Panel indices MODEL_INFO(0), TIMING(1), EPOCH_HISTORY(2), RESULTS(3)
    // are scrollable.  PROGRESS(4) is not.
    if (this->activePanel >= 0 && this->activePanel < 4) {
      TerminalUI_Panel* active = scrollablePanels[this->activePanel];

      if (active && active->applyScrollInput(ch))
        return true;
    }

    return false;
  }

  //===================================================================================================================//
  //-- Widget overrides --//
  //===================================================================================================================//

  bool TerminalUI_PredictWindow::handleEvent(int ch)
  {
    // Dismiss keys: 'q', 'Q', ESC (27)
    if (ch == 'q' || ch == 'Q' || ch == 27) {
      this->dismissed.store(true);
      return true;
    }

    if (this->cycleActivePanel(ch))
      return true;

    if (this->scrollActivePanel(ch))
      return true;

    // Non-scroll events propagate to all children.
    return TerminalUI_Window::handleEvent(ch);
  }

  //===================================================================================================================//
  //-- Private — progress bar layout sync --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::syncProgressBarLayout()
  {
    if (!this->loadingBarPtr || !this->progressBarPtr)
      return;

    // Grant both bars the widest label and per-segment suffix either of them
    // needs so their brackets stay vertically aligned (comparable progress)
    // while the bars themselves take every remaining column.
    int labelReserve = std::max(this->loadingBarPtr->requiredLabelWidth(), this->progressBarPtr->requiredLabelWidth());
    int suffixReserve =
      std::max(this->loadingBarPtr->requiredSuffixWidth(), this->progressBarPtr->requiredSuffixWidth());

    this->loadingBarPtr->setReservedLabelWidth(labelReserve);
    this->progressBarPtr->setReservedLabelWidth(labelReserve);
    this->loadingBarPtr->setReservedSuffixWidth(suffixReserve);
    this->progressBarPtr->setReservedSuffixWidth(suffixReserve);
  }

  //===================================================================================================================//
  //-- Protected — helpers --//
  //===================================================================================================================//

  void TerminalUI_PredictWindow::updatePanelColors()
  {
    // Active panel: YELLOW (3).  Inactive panels: CYAN (2).
    // The progress panel is always CYAN since it is not scrollable.
    if (this->progressPanelPtr)
      this->progressPanelPtr->setColorPair(2);

    if (this->modelInfoPanelPtr)
      this->modelInfoPanelPtr->setColorPair(this->activePanel == MODEL_INFO ? 3 : 2);

    if (this->timingPanelPtr)
      this->timingPanelPtr->setColorPair(this->activePanel == TIMING ? 3 : 2);

    if (this->epochHistoryPanelPtr)
      this->epochHistoryPanelPtr->setColorPair(this->activePanel == EPOCH_HISTORY ? 3 : 2);

    if (this->resultsPanelPtr)
      this->resultsPanelPtr->setColorPair(this->activePanel == RESULTS ? 3 : 2);
  }

} // namespace NN_CLI
