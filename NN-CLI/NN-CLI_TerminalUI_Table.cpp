#include "NN-CLI_TerminalUI_Table.hpp"

#include <algorithm>

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TerminalUI_Table::TerminalUI_Table() {}

  //===================================================================================================================//

  TerminalUI_Table::TerminalUI_Table(int maxWidth)
    : maxWidth(maxWidth)
  {
  }

  //===================================================================================================================//

  TerminalUI_Table::TerminalUI_Table(std::vector<Column> columns, int maxWidth)
    : columns(std::move(columns)),
      maxWidth(maxWidth)
  {
  }

  //===================================================================================================================//
  //-- Configuration --//
  //===================================================================================================================//

  void TerminalUI_Table::setColumns(std::vector<Column> columns)
  {
    this->columns = std::move(columns);
    this->widthsDirty = true;
  }

  //===================================================================================================================//

  void TerminalUI_Table::setTitle(const std::string& title)
  {
    this->title = title;
  }

  //===================================================================================================================//

  void TerminalUI_Table::setMaxWidth(int maxWidth)
  {
    this->maxWidth = maxWidth;
    this->widthsDirty = true;
  }

  //===================================================================================================================//
  //-- Data --//
  //===================================================================================================================//

  void TerminalUI_Table::addRow(const Row& cells)
  {
    this->rows.push_back(cells);
  }

  //===================================================================================================================//

  void TerminalUI_Table::addRows(const std::vector<Row>& rows)
  {
    this->rows.insert(this->rows.end(), rows.begin(), rows.end());
  }

  //===================================================================================================================//

  void TerminalUI_Table::clearRows()
  {
    this->rows.clear();
  }

  //===================================================================================================================//
  //-- Rendering --//
  //===================================================================================================================//

  std::vector<std::string> TerminalUI_Table::render() const
  {
    if (this->widthsDirty)
      this->computeWidths();

    std::vector<std::string> lines;

    if (this->columns.empty())
      return lines;

    // Top border
    lines.push_back(this->makeSeparator());

    // Title row (spans all columns, centered) — replaces the header row.
    if (!this->title.empty()) {
      lines.push_back(this->makeTitleRow());
      lines.push_back(this->makeSeparator());
    } else {
      // Header row with column names
      Row headers;
      for (const auto& col : this->columns)
        headers.push_back(col.name);

      lines.push_back(this->makeRow(headers));
      lines.push_back(this->makeSeparator());
    }

    // Data rows (one empty placeholder row if no data has been added)
    if (this->rows.empty()) {
      Row emptyRow(this->columns.size(), std::string());
      lines.push_back(this->makeRow(emptyRow));
    } else {
      for (const auto& row : this->rows)
        lines.push_back(this->makeRow(row));
    }

    // Bottom border
    lines.push_back(this->makeSeparator());

    return lines;
  }

  //===================================================================================================================//
  //-- Accessors --//
  //===================================================================================================================//

  int TerminalUI_Table::getMaxWidth() const
  {
    return this->maxWidth;
  }

  //===================================================================================================================//

  int TerminalUI_Table::columnCount() const
  {
    return static_cast<int>(this->columns.size());
  }

  //===================================================================================================================//

  const std::vector<int>& TerminalUI_Table::getComputedWidths() const
  {
    if (this->widthsDirty)
      this->computeWidths();

    return this->computedWidths;
  }

  //===================================================================================================================//

  std::string TerminalUI_Table::separator() const
  {
    if (this->widthsDirty)
      this->computeWidths();

    return this->makeSeparator();
  }

  //===================================================================================================================//
  //-- Private — width computation --//
  //===================================================================================================================//

  void TerminalUI_Table::computeWidths() const
  {
    const int N = static_cast<int>(this->columns.size());

    if (N == 0) {
      this->computedWidths.clear();
      this->widthsDirty = false;
      return;
    }

    // Structural overhead per row: "| " (2) + " | " (3) * (N-1) + " |" (2) = 3*N + 1.
    // The separator line has the same total length: "+" (1) + ("---...-+") * N
    // where each segment is (width+2) dashes plus one "+" = (width+3),
    // giving 1 + sum(width+3) = 1 + sum(width) + 3*N = 3*N + 1 + sum(width).
    // So dataWidth + overhead = row total length = maxWidth.
    const int overhead = 3 * N + 1;
    const int dataWidth = std::max(0, this->maxWidth - overhead);

    // Sum width hints for proportional allocation.
    int totalHints = 0;
    for (const auto& col : this->columns)
      totalHints += std::max(1, col.widthHint);

    this->computedWidths.resize(N);

    if (dataWidth <= 0 || totalHints <= 0) {
      // No usable space or zero hints — equal distribution as fallback.
      const int perCol = dataWidth > 0 ? dataWidth / N : 0;
      for (int i = 0; i < N; i++)
        this->computedWidths[i] = perCol;

      this->widthsDirty = false;
      return;
    }

    // Proportional allocation: distribute dataWidth by weight.
    // The last column absorbs rounding remainder so the row fills maxWidth exactly.
    int allocated = 0;
    for (int i = 0; i < N - 1; i++) {
      int w = dataWidth * std::max(1, this->columns[i].widthHint) / totalHints;
      w = std::max(1, w);
      this->computedWidths[i] = w;
      allocated += w;
    }

    this->computedWidths[N - 1] = std::max(1, dataWidth - allocated);

    this->widthsDirty = false;
  }

  //===================================================================================================================//
  //-- Private — cell formatting --//
  //===================================================================================================================//

  std::string TerminalUI_Table::formatCell(const std::string& text, int width, Align align) const
  {
    if (width <= 0)
      return text;

    const int textLen = static_cast<int>(text.size());
    const int padding = std::max(0, width - textLen);

    switch (align) {
    case Align::LEFT:
      return text + std::string(padding, ' ');
    case Align::RIGHT:
      return std::string(padding, ' ') + text;
    case Align::CENTER: {
      const int leftPad = padding / 2;
      const int rightPad = padding - leftPad;
      return std::string(leftPad, ' ') + text + std::string(rightPad, ' ');
    }
    }

    return text;
  }

  //===================================================================================================================//
  //-- Private — row / separator builders --//
  //===================================================================================================================//

  std::string TerminalUI_Table::makeRow(const Row& cells) const
  {
    std::string result = "|";

    for (int i = 0; i < static_cast<int>(this->columns.size()); i++) {
      std::string cellText = (i < static_cast<int>(cells.size())) ? cells[i] : "";

      // Apply per-column formatter if present.
      if (this->columns[i].formatter)
        cellText = this->columns[i].formatter(cellText);

      result += " " + this->formatCell(cellText, this->computedWidths[i], this->columns[i].align) + " |";
    }

    return result;
  }

  //===================================================================================================================//

  std::string TerminalUI_Table::makeSeparator() const
  {
    std::string result = "+";

    for (int i = 0; i < static_cast<int>(this->computedWidths.size()); i++)
      result += std::string(this->computedWidths[i] + 2, '-') + "+";

    return result;
  }

  //===================================================================================================================//

  std::string TerminalUI_Table::makeTitleRow() const
  {
    const int N = static_cast<int>(this->computedWidths.size());

    // Total content width between the outer "|" markers of a regular row.
    // A regular row: "| " + w0 + " | " + w1 + ... + " | " + wN-1 + " |"
    // Between outer "|"s: " w0 | w1 | ... | wN-1 " = sum(wi) + 3*N - 1
    int titleSpace = 3 * N - 1;
    for (int w : this->computedWidths)
      titleSpace += w;

    const int titleLen = static_cast<int>(this->title.size());
    const int padding = std::max(0, titleSpace - titleLen);
    const int leftPad = padding / 2;
    const int rightPad = padding - leftPad;

    return "|" + std::string(leftPad, ' ') + this->title + std::string(rightPad, ' ') + "|";
  }

} // namespace NN_CLI
