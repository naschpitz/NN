#include "NN-CLI_TerminalUI_ProgressBar.hpp"

#include <algorithm>
#include <curses.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
  //-- Layout constants --//
  constexpr int kLabelWidth = 28; // legacy WINDOW* renderers only: labels left-padded to this width
  constexpr int kBracketWidth = 2; // the " [" written right after the label

  //-- Color pairs (configured in TerminalUI/Window init). --//
  constexpr int kBarColor = 1; // green filled bar segment

  //-- Percentage format shared by rendering and width reservation. --//
  constexpr const char* kPctFormat = "] %6.3f%%";

  //===================================================================================================================//
  // Width of the percentage suffix at its widest ("] 100.000%"), measured from
  // the actual format so the layout reservation can never drift from what is
  // rendered.
  int pctWidth()
  {
    static const int width = [] {
      char buf[16];
      return snprintf(buf, sizeof(buf), kPctFormat, 100.0);
    }();

    return width;
  }

  //===================================================================================================================//
  // Usable bar width inside `win`, with `suffixCols` reserved on the right for
  // the per-segment suffix.
  int barWidthFor(int cols, int suffixCols)
  {
    return std::max(10, cols - kLabelWidth - kBracketWidth - pctWidth() - suffixCols);
  }

  //===================================================================================================================//
  // Usable bar width when drawing on stdscr, with `labelCols` granted on the
  // left for the label and `suffixCols` reserved on the right for the
  // per-segment suffix.
  int barWidthForStdscr(int width, int labelCols, int suffixCols)
  {
    int cols = width > 0 ? width : 80;
    return std::max(10, cols - labelCols - kBracketWidth - pctWidth() - suffixCols);
  }

  //===================================================================================================================//
  // Emit the label left-padded to `labelCols`, then " [", onto stdscr at (row, col).
  void emitLabelStdscr(int row, int col, const std::string& label, int labelCols)
  {
    int len = static_cast<int>(label.size());
    mvaddstr(row, col, label.c_str());

    for (int i = len; i < labelCols; i++)
      mvaddch(row, col + i, ' ');

    mvaddstr(row, col + labelCols, " [");
  }

  //===================================================================================================================//
  // Emit the label left-padded to kLabelWidth, then " [".
  void emitLabel(WINDOW* win, const std::string& label)
  {
    int len = static_cast<int>(label.size());
    ::waddstr(win, label.c_str());

    for (int i = len; i < kLabelWidth; i++)
      ::waddstr(win, " ");

    ::waddstr(win, " [");
  }

  //===================================================================================================================//
  // Emit a single solid bar on stdscr starting at absolute column `startCol`:
  // `filled` green blocks, then light shading out to `barWidth`.
  // The block glyphs are multi-byte UTF-8, so they must go through mvaddstr —
  // mvaddch takes a chtype and would truncate them to garbage.
  void emitSingleBarStdscr(int row, int startCol, float fraction, int barWidth)
  {
    int filled = std::clamp(static_cast<int>(fraction * barWidth), 0, barWidth);

    attron(COLOR_PAIR(kBarColor));

    for (int i = 0; i < filled; i++)
      mvaddstr(row, startCol + i, "\xe2\x96\x88");

    attroff(COLOR_PAIR(kBarColor));

    for (int i = filled; i < barWidth; i++)
      mvaddstr(row, startCol + i, "\xe2\x96\x91");
  }

  //===================================================================================================================//
  // Emit a single solid bar: `filled` green blocks, then light shading out to `barWidth`.
  void emitSingleBar(WINDOW* win, float fraction, int barWidth)
  {
    int filled = std::clamp(static_cast<int>(fraction * barWidth), 0, barWidth);

    ::wattron(win, COLOR_PAIR(kBarColor));

    for (int i = 0; i < filled; i++)
      ::waddstr(win, "\xe2\x96\x88");

    ::wattroff(win, COLOR_PAIR(kBarColor));

    for (int i = filled; i < barWidth; i++)
      ::waddstr(win, "\xe2\x96\x91");
  }

  //===================================================================================================================//
  // Emit a segmented bar on stdscr starting at absolute column `startCol`, one
  // segment per fraction separated by "│".  Segments share barWidth minus the
  // separator columns; the last segment absorbs the division remainder so the
  // bar fills barWidth exactly.
  void emitSegmentedBarStdscr(int row, int startCol, const std::vector<float>& fractions, int numSegments,
                              int barWidth)
  {
    int separators = numSegments - 1;
    int segmentWidth = (barWidth - separators) / numSegments;
    int lastSegmentWidth = (barWidth - separators) - segmentWidth * separators;

    for (int seg = 0; seg < numSegments; seg++) {
      float pct = (seg < static_cast<int>(fractions.size())) ? fractions[seg] : 0.0f;
      int width = (seg == separators) ? lastSegmentWidth : segmentWidth;
      emitSingleBarStdscr(row, startCol, pct, width);
      startCol += width;

      if (seg < separators) {
        mvaddstr(row, startCol, "\xe2\x94\x82");
        startCol += 1;
      }
    }
  }

  //===================================================================================================================//
  // Emit a segmented bar, one segment per fraction separated by "│".
  // Same width distribution as emitSegmentedBarStdscr.
  void emitSegmentedBar(WINDOW* win, const std::vector<float>& fractions, int numSegments, int barWidth)
  {
    int separators = numSegments - 1;
    int segmentWidth = (barWidth - separators) / numSegments;
    int lastSegmentWidth = (barWidth - separators) - segmentWidth * separators;

    for (int seg = 0; seg < numSegments; seg++) {
      float pct = (seg < static_cast<int>(fractions.size())) ? fractions[seg] : 0.0f;
      emitSingleBar(win, pct, (seg == separators) ? lastSegmentWidth : segmentWidth);

      if (seg < separators)
        ::waddstr(win, "\xe2\x94\x82");
    }
  }

  //===================================================================================================================//
  // Build the trailing per-segment suffix " (0:XX% | 1:XX%)".
  std::string buildSegmentSuffix(const std::vector<float>& fractions)
  {
    std::ostringstream info;
    info << " (";

    for (size_t s = 0; s < fractions.size(); s++) {
      info << s << ":" << std::setw(3) << static_cast<int>(fractions[s] * 100) << "%";

      if (s + 1 < fractions.size())
        info << " | ";
    }

    info << ")";
    return info.str();
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TerminalUI_ProgressBar::TerminalUI_ProgressBar() {}

  //===================================================================================================================//
  //-- Widget data --//
  //===================================================================================================================//

  void TerminalUI_ProgressBar::setBarData(const std::string& label, float fraction)
  {
    this->barLabel = label;
    this->barFractions = {std::clamp(fraction, 0.0f, 1.0f)};
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::setBarData(const std::string& label, const std::vector<float>& fractions)
  {
    this->barLabel = label;
    this->barFractions.clear();
    this->barFractions.reserve(fractions.size());

    for (float f : fractions)
      this->barFractions.push_back(std::clamp(f, 0.0f, 1.0f));
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::setSubLineText(const std::string& text, int colorPair)
  {
    this->subLineText = text;
    this->subLineColorPair = colorPair;
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::clearSubLineText()
  {
    this->subLineText.clear();
    this->subLineColorPair = 0;
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::setVisible(bool visible)
  {
    this->visible = visible;
  }

  //===================================================================================================================//

  int TerminalUI_ProgressBar::requiredLabelWidth() const
  {
    return static_cast<int>(this->barLabel.size());
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::setReservedLabelWidth(int cols)
  {
    // Reservations only grow so bar geometry stays stable across transient
    // states (e.g. a shorter label while a counter has fewer digits).
    this->reservedLabelWidth = std::max(this->reservedLabelWidth, cols);
  }

  //===================================================================================================================//

  int TerminalUI_ProgressBar::requiredSuffixWidth() const
  {
    if (this->barFractions.size() <= 1)
      return 0;

    // Measure the actual formatted suffix: setw keeps its width constant for
    // a given segment count, so this is stable while values change and scales
    // with any number of segments.
    return static_cast<int>(buildSegmentSuffix(this->barFractions).size());
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::setReservedSuffixWidth(int cols)
  {
    // Reservations only grow so bar geometry stays stable across transient
    // states (e.g. the single-segment "Validating" bar between epochs).
    this->reservedSuffixWidth = std::max(this->reservedSuffixWidth, cols);
  }

  //===================================================================================================================//
  //-- Widget overrides --//
  //===================================================================================================================//

  void TerminalUI_ProgressBar::draw()
  {
    if (!this->visible || this->width <= 0 || this->height <= 0)
      return;

    int numSegments = std::max(1, static_cast<int>(this->barFractions.size()));

    // Compute the fraction to display (average for multi-segment).
    float fraction = 0.0f;

    for (float f : this->barFractions)
      fraction += f;

    if (!this->barFractions.empty())
      fraction /= static_cast<float>(this->barFractions.size());

    // Grant at least this bar's own measured label and suffix widths; the
    // externally granted reservations (synced across sibling bars by the
    // parent) can widen them so all bars share identical bracket positions
    // while the bar itself takes every remaining column.
    std::string suffix = (numSegments > 1) ? buildSegmentSuffix(this->barFractions) : std::string();
    int suffixCols = std::max(this->reservedSuffixWidth, static_cast<int>(suffix.size()));
    int labelCols = std::max(this->reservedLabelWidth, static_cast<int>(this->barLabel.size()));
    int bw = barWidthForStdscr(this->width, labelCols, suffixCols);
    int barStart = this->x + labelCols + kBracketWidth;

    // Draw label and bar on the first row.
    emitLabelStdscr(this->y, this->x, this->barLabel, labelCols);

    if (numSegments == 1) {
      emitSingleBarStdscr(this->y, barStart, this->barFractions.empty() ? 0.0f : this->barFractions[0], bw);
    } else {
      emitSegmentedBarStdscr(this->y, barStart, this->barFractions, numSegments, bw);
    }

    // Percentage suffix.
    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), kPctFormat, static_cast<double>(fraction * 100.0f));
    int pctCol = barStart + bw;
    mvaddstr(this->y, pctCol, pctBuf);

    // Per-segment suffix " (0:XX% | 1:XX%)" right after the percentage, when it fits.
    if (!suffix.empty()) {
      int suffixCol = pctCol + static_cast<int>(strlen(pctBuf));

      if (suffixCol + static_cast<int>(suffix.size()) <= this->x + this->width)
        mvaddstr(this->y, suffixCol, suffix.c_str());
    }

    // Draw sub-line on the second row (if present).
    if (this->height >= 2 && !this->subLineText.empty()) {
      if (this->subLineColorPair > 0)
        attron(COLOR_PAIR(this->subLineColorPair));

      mvaddstr(this->y + 1, this->x, this->subLineText.c_str());

      if (this->subLineColorPair > 0)
        attroff(COLOR_PAIR(this->subLineColorPair));

      // Clear stale content beyond the text.
      int textLen = static_cast<int>(this->subLineText.size());
      int maxClear = this->width - (this->x + textLen);

      for (int i = 0; i < maxClear; i++)
        mvaddch(this->y + 1, this->x + textLen + i, ' ');
    }
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::resize(int width, int height, int x, int y)
  {
    TerminalUI_Widget::resize(width, height, x, y);
  }

  //===================================================================================================================//
  //-- ncurses Rendering (legacy) --//
  //===================================================================================================================//

  void TerminalUI_ProgressBar::renderSingleBar(WINDOW* win, const std::string& label, float fraction)
  {
    if (!win)
      return;

    fraction = std::clamp(fraction, 0.0f, 1.0f);

    ::werase(win);

    int bw = barWidthFor(::getmaxx(win), 0);
    emitLabel(win, label);
    emitSingleBar(win, fraction, bw);

    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), kPctFormat, static_cast<double>(fraction * 100.0f));
    ::waddstr(win, pctBuf);

    ::wnoutrefresh(win);
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::renderMultiBar(WINDOW* win, const std::string& label,
                                              const std::vector<float>& fractions)
  {
    if (!win)
      return;

    int numSegments = std::max(1, static_cast<int>(fractions.size()));

    // Clamp all fractions to [0.0, 1.0].
    std::vector<float> clamped;
    clamped.reserve(fractions.size());

    for (float f : fractions)
      clamped.push_back(std::clamp(f, 0.0f, 1.0f));

    ::werase(win);

    int cols = ::getmaxx(win);
    std::string suffix = (numSegments > 1) ? buildSegmentSuffix(clamped) : std::string();
    int bw = barWidthFor(cols, static_cast<int>(suffix.size()));

    emitLabel(win, label);
    emitSegmentedBar(win, clamped, numSegments, bw);

    // Compute average percentage across all segments.
    float totalPct = 0.0f;

    for (float f : clamped)
      totalPct += f;

    float avgPct = clamped.empty() ? 0.0f : (totalPct / static_cast<float>(clamped.size())) * 100.0f;

    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), kPctFormat, static_cast<double>(avgPct));
    ::waddstr(win, pctBuf);

    // Emit the per-segment suffix only when it fits within the window.
    int overhead = kLabelWidth + kBracketWidth + pctWidth() + static_cast<int>(suffix.size());

    if (!suffix.empty() && bw + overhead <= cols)
      ::waddstr(win, suffix.c_str());

    ::wnoutrefresh(win);
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::renderSubLine(WINDOW* win, const std::string& text, int colorPair)
  {
    if (!win)
      return;

    ::wmove(win, 1, 0);

    if (colorPair > 0)
      ::wattron(win, COLOR_PAIR(colorPair));

    ::waddstr(win, text.c_str());

    if (colorPair > 0)
      ::wattroff(win, COLOR_PAIR(colorPair));

    // Clear any stale content beyond the new text.
    ::wclrtoeol(win);

    ::wnoutrefresh(win);
  }

  //===================================================================================================================//

  void TerminalUI_ProgressBar::clearSubLine(WINDOW* win)
  {
    if (!win)
      return;

    ::wmove(win, 1, 0);
    ::wclrtoeol(win);
  }

  //===================================================================================================================//
  //-- stdout Rendering --//
  //===================================================================================================================//

  void TerminalUI_ProgressBar::printLoadingProgress(const std::string& label, size_t current, size_t total,
                                                    ulong progressReports, int barWidth)
  {
    ulong interval = (progressReports > 0) ? std::max(static_cast<size_t>(1), total / progressReports) : 0;

    if (interval == 0)
      return;

    if (current > 1 && current != total && (current % interval) != 0)
      return;

    float percent = (total > 0) ? static_cast<float>(current) / static_cast<float>(total) : 0.0f;
    percent = std::clamp(percent, 0.0f, 1.0f);
    int filledWidth = static_cast<int>(percent * barWidth);

    std::ostringstream out;
    out << "\r" << label << " [";

    for (int i = 0; i < barWidth; i++)
      out << (i < filledWidth ? "\xe2\x96\x88" : "\xe2\x96\x91");

    out << "] " << current << "/" << total << "  " << std::fixed << std::setprecision(1) << (percent * 100.0f) << "%";
    out << "   ";

    std::cout << out.str() << std::flush;

    if (current == total)
      std::cout << std::endl;
  }

} // namespace NN_CLI
