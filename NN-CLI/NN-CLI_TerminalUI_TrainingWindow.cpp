#include "NN-CLI_TerminalUI_TrainingWindow.hpp"

#include <algorithm>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors / Dtors --//
  //===================================================================================================================//

  TerminalUI_TrainingWindow::TerminalUI_TrainingWindow()
  {
    //-- Create and register child panels --//

    auto progressPanel = std::make_unique<TerminalUI_Panel>("Progress", 2);
    auto epochsPanel = std::make_unique<TerminalUI_Panel>("Epochs", 2);
    auto modelInfoPanel = std::make_unique<TerminalUI_Panel>("Model Info", 2);
    auto timingPanel = std::make_unique<TerminalUI_Panel>("Timing", 2);

    this->progressPanelPtr = progressPanel.get();
    this->epochsPanelPtr = epochsPanel.get();
    this->modelInfoPanelPtr = modelInfoPanel.get();
    this->timingPanelPtr = timingPanel.get();

    this->addChild(std::move(progressPanel));
    this->addChild(std::move(epochsPanel));
    this->addChild(std::move(modelInfoPanel));
    this->addChild(std::move(timingPanel));

    //-- Configure panels --//

    this->epochsPanelPtr->setAutoScroll(true);

    //-- Create and attach the two stacked progress bars inside the progress --//
    //-- panel: the "Samples" loading bar on top, the training bar below.     --//

    auto loadingBar = std::make_unique<TerminalUI_ProgressBar>();
    this->loadingBarPtr = loadingBar.get();
    this->loadingBarPtr->setVisible(false); // hidden until the first sample load
    this->progressPanelPtr->addChild(std::move(loadingBar));

    auto progressBar = std::make_unique<TerminalUI_ProgressBar>();
    this->progressBarPtr = progressBar.get();
    this->progressPanelPtr->addChild(std::move(progressBar));

    //-- Configure the epoch table with default columns --//

    this->epochTable.setColumns({
      {"Epoch", 5, TerminalUI_Table::Align::RIGHT},
      {"Loss", 8, TerminalUI_Table::Align::RIGHT},
      {"Accuracy", 8, TerminalUI_Table::Align::RIGHT},
      {"Best", 4, TerminalUI_Table::Align::LEFT},
      {"Completed At", 19, TerminalUI_Table::Align::LEFT},
    });
  }

  //===================================================================================================================//

  TerminalUI_TrainingWindow::~TerminalUI_TrainingWindow() {}

  //===================================================================================================================//
  //-- Layout --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::layoutChildren()
  {
    int W = this->width;
    int H = this->height;

    if (W <= 0 || H <= 0)
      return;

    //-- Reserve the progress panel at the bottom --//

    int progressH = std::min(kProgressHeight, H);
    int remainingH = std::max(0, H - progressH);

    //-- Horizontal split: timing panel on the right if screen is wide enough --//

    int timingW = 0;
    int leftW = W;

    if (W >= kMinLeftWidth + kMinTimingWidth) {
      int idealTimingW = W * 35 / 100;
      timingW = std::max(kMinTimingWidth, std::min(W - kMinLeftWidth, idealTimingW));
      leftW = W - timingW;
    }

    //-- Vertical split of the left column: model info ~45%, epochs ~55% --//

    int modelInfoH = std::max(3, remainingH * 45 / 100);
    int epochsH = std::max(3, remainingH - modelInfoH);

    //-- Position panels --//
    //   modelInfoPanel  — top-left:       (0, 0)            size: (leftW, modelInfoH)
    //   epochsPanel     — bottom-left:    (0, modelInfoH)   size: (leftW, epochsH)
    //   timingPanel     — right column:   (leftW, 0)        size: (timingW, remainingH)
    //   progressPanel   — bottom full:    (0, H - progressH) size: (W, progressH)

    if (this->childCount() < 4)
      return;

    this->children[0]->resize(W, progressH, 0, H - progressH);
    this->children[1]->resize(leftW, epochsH, 0, modelInfoH);
    this->children[2]->resize(leftW, modelInfoH, 0, 0);

    if (timingW > 0)
      this->children[3]->resize(timingW, remainingH, leftW, 0);
    else
      this->children[3]->resize(0, 0, 0, 0);

    //-- Stack the two progress bars inside the progress panel --//
    // The generic Panel layout places all children at the content origin (so
    // they would overlap); reposition them onto separate rows here: the
    // "Samples" loading bar on top (1 row) and the training bar below it
    // (remaining rows = bar + sub-line).
    int contentX = 2;             // progress panel x (0) + border/pad
    int contentY = (H - progressH) + 1; // progress panel y + top border
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
    this->refreshEpochContent();
    this->refreshModelInfoContent();
    this->refreshTimingContent();
  }

  //===================================================================================================================//
  //-- Progress updates --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::updateProgress(const std::string& label, float fraction)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setBarData(label, fraction);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::updateProgress(const std::string& label,
                                                  const std::vector<float>& fractions)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setBarData(label, fractions);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::updateProgressSubLine(const std::string& text, int colorPair)
  {
    if (this->progressBarPtr)
      this->progressBarPtr->setSubLineText(text, colorPair);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearProgressSubLine()
  {
    if (this->progressBarPtr)
      this->progressBarPtr->clearSubLineText();
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setLoadingProgress(const std::string& label, float fraction)
  {
    if (this->loadingBarPtr) {
      this->loadingBarPtr->setVisible(true);
      this->loadingBarPtr->setBarData(label, fraction);
    }
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearLoadingProgress()
  {
    if (this->loadingBarPtr)
      this->loadingBarPtr->setVisible(false);
  }

  //===================================================================================================================//
  //-- Epoch table --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setEpochColumns(std::vector<TerminalUI_Table::Column> columns)
  {
    this->epochTable.setColumns(std::move(columns));
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::addEpochRow(const TerminalUI_Table::Row& row)
  {
    this->epochTable.addRow(row);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::addEpochRows(const std::vector<TerminalUI_Table::Row>& rows)
  {
    this->epochTable.addRows(rows);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearEpochRows()
  {
    this->epochTable.clearRows();
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::addEpochMessage(const std::string& message)
  {
    this->epochMessages.push_back(message);
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearEpochMessages()
  {
    this->epochMessages.clear();
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::refreshEpochContent()
  {
    if (!this->epochsPanelPtr)
      return;

    // Compute the effective table width from the panel's content area.
    // contentWidth() depends on whether the scrollbar is shown, which in turn
    // depends on the number of lines currently stored in the panel.  After
    // rendering and setting new lines the scrollbar status may flip, changing
    // contentWidth by 1 column.  Detect that transition and re-render at the
    // correct width so the table fills the content area precisely.
    int tableWidth = std::max(40, this->epochsPanelPtr->contentWidth());
    this->epochTable.setMaxWidth(tableWidth);

    // Render the table to formatted lines.
    auto lines = this->epochTable.render();

    // Append status / monitor messages below the table.
    for (const auto& msg : this->epochMessages)
      lines.push_back(msg);

    this->epochsPanelPtr->setLines(lines);

    // If the new line count toggled the scrollbar, contentWidth() may have
    // changed.  Re-render at the updated width so column alignment is exact.
    int revisedWidth = std::max(40, this->epochsPanelPtr->contentWidth());

    if (revisedWidth != tableWidth) {
      this->epochTable.setMaxWidth(revisedWidth);
      lines = this->epochTable.render();

      for (const auto& msg : this->epochMessages)
        lines.push_back(msg);

      this->epochsPanelPtr->setLines(lines);
    }
  }

  //===================================================================================================================//
  //-- Model info --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setModelInfoTitle(const std::string& title)
  {
    this->modelInfoTitle = title;
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setModelInfoEntries(
    const std::vector<std::pair<std::string, std::string>>& entries)
  {
    this->modelConfigRows.clear();

    for (const auto& entry : entries)
      this->modelConfigRows.push_back({entry.first, entry.second});
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::addModelInfoEntry(const std::string& key, const std::string& value)
  {
    this->modelConfigRows.push_back({key, value});
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setModelInfoRows(const std::vector<SummaryRow>& rows)
  {
    this->modelConfigRows = rows;
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearModelInfoEntries()
  {
    this->modelConfigRows.clear();
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setLossReferenceRows(const std::vector<SummaryRow>& rows)
  {
    this->lossReferenceRows = rows;
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::clearLossReferenceRows()
  {
    this->lossReferenceRows.clear();
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::refreshModelInfoContent()
  {
    if (!this->modelInfoPanelPtr)
      return;

    int tableWidth = std::max(30, this->modelInfoPanelPtr->contentWidth());

    std::vector<SummaryTable::Section> sections;

    SummaryTable::Section configSection;
    configSection.title = this->modelInfoTitle.empty() ? "Model Configuration" : this->modelInfoTitle;
    configSection.rows = this->modelConfigRows;
    sections.push_back(std::move(configSection));

    if (!this->lossReferenceRows.empty()) {
      SummaryTable::Section lossSection;
      lossSection.title = "Loss Reference";
      lossSection.rows = this->lossReferenceRows;
      sections.push_back(std::move(lossSection));
    }

    auto lines = SummaryTable::collectSections(sections, static_cast<ulong>(tableWidth));
    this->modelInfoPanelPtr->setLines(lines);
    this->modelInfoPanelPtr->scrollState().offset = 0;
  }

  //===================================================================================================================//
  //-- Timing --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setTimingLines(const std::vector<std::string>& lines)
  {
    this->rawTimingLines = lines;
  }

  //===================================================================================================================//

  void TerminalUI_TrainingWindow::refreshTimingContent()
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
  //-- Panel selection --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::setActivePanel(int panelIndex)
  {
    this->activePanel = panelIndex;
  }

  //===================================================================================================================//

  int TerminalUI_TrainingWindow::getActivePanel() const
  {
    return this->activePanel;
  }

  //===================================================================================================================//
  //-- Panel access --//
  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_TrainingWindow::getProgressPanel() const
  {
    return this->progressPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_TrainingWindow::getEpochsPanel() const
  {
    return this->epochsPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_TrainingWindow::getModelInfoPanel() const
  {
    return this->modelInfoPanelPtr;
  }

  //===================================================================================================================//

  TerminalUI_Panel* TerminalUI_TrainingWindow::getTimingPanel() const
  {
    return this->timingPanelPtr;
  }

  //===================================================================================================================//
  //-- Hooks --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::preRender()
  {
    this->updatePanelColors();
  }

  //===================================================================================================================//
  //-- Event routing --//
  //===================================================================================================================//

  bool TerminalUI_TrainingWindow::cycleActivePanel(int ch)
  {
    if (ch != '\t')
      return false;

    this->activePanel = (this->activePanel + 1) % 3;
    return true;
  }

  //===================================================================================================================//

  bool TerminalUI_TrainingWindow::scrollActivePanel(int ch)
  {
    TerminalUI_Panel* scrollablePanels[] = {
      this->modelInfoPanelPtr,
      this->epochsPanelPtr,
      this->timingPanelPtr,
    };

    if (this->activePanel >= 0 && this->activePanel < 3) {
      TerminalUI_Panel* active = scrollablePanels[this->activePanel];

      if (active && active->applyScrollInput(ch))
        return true;
    }

    return false;
  }

  //===================================================================================================================//
  //-- Widget overrides --//
  //===================================================================================================================//

  bool TerminalUI_TrainingWindow::handleEvent(int ch)
  {
    if (this->cycleActivePanel(ch))
      return true;

    if (this->scrollActivePanel(ch))
      return true;

    // Non-scroll events propagate to all children.
    return TerminalUI_Window::handleEvent(ch);
  }

  //===================================================================================================================//
  //-- Protected — helpers --//
  //===================================================================================================================//

  void TerminalUI_TrainingWindow::updatePanelColors()
  {
    // Active panel: YELLOW (3).  Inactive panels: CYAN (2).
    // The progress panel is always CYAN since it is not scrollable.
    if (this->progressPanelPtr)
      this->progressPanelPtr->setColorPair(2);

    if (this->modelInfoPanelPtr)
      this->modelInfoPanelPtr->setColorPair(this->activePanel == MODEL_INFO ? 3 : 2);

    if (this->epochsPanelPtr)
      this->epochsPanelPtr->setColorPair(this->activePanel == EPOCHS ? 3 : 2);

    if (this->timingPanelPtr)
      this->timingPanelPtr->setColorPair(this->activePanel == TIMING ? 3 : 2);
  }

} // namespace NN_CLI
