#include "NN-CLI_TerminalUI_ProgressBar.hpp"

#include <algorithm>
#include <curses.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
  //-- Layout constants shared by every ncurses progress bar so they all line up
  //-- vertically in the Training panel. --//
  constexpr int kLabelWidth = 28; // labels are left-padded to this width
  constexpr int kBracketWidth = 2; // the " [" written right after the label
  constexpr int kRightPad = 20; // reserved on the right for "] 100.000%" suffix (+ slack)
  constexpr int kSegmentInfoPerSegment = 9; // width of one " | N:XXX%" cell in the per-segment suffix

  //-- Color pairs (configured in TerminalUI/Window init). --//
  constexpr int kBarColor = 1; // green filled bar segment

  //===================================================================================================================//
  // Width reserved on the right for the per-segment "(0:XX% | 1:XX%)" suffix (0 when single-segment).
  int segmentSuffixWidth(int segments)
  {
    return segments > 1 ? kSegmentInfoPerSegment * segments - 1 : 0;
  }

  //===================================================================================================================//
  // Usable bar width inside `win`, mirrored across all bars so their brackets line up.
  int barWidthFor(int cols, int segments)
  {
    return std::max(10, cols - kLabelWidth - kBracketWidth - kRightPad - segmentSuffixWidth(segments));
  }

  //===================================================================================================================//
  // Usable bar width when drawing on stdscr at a given x position.
  int barWidthForStdscr(int x, int width, int segments)
  {
    int cols = width > 0 ? width : 80;
    return std::max(10, cols - kLabelWidth - kBracketWidth - kRightPad - segmentSuffixWidth(segments));
  }

  //===================================================================================================================//
  // Emit the label left-padded to kLabelWidth, then " [", onto stdscr at (row, col).
  void emitLabelStdscr(int row, int col, const std::string& label)
  {
    int len = static_cast<int>(label.size());
    mvaddstr(row, col, label.c_str());

    for (int i = len; i < kLabelWidth; i++)
      mvaddch(row, col + i, ' ');

    mvaddstr(row, col + kLabelWidth, " [");
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
  // Emit a segmented bar on stdscr, one segment per fraction separated by "│".
  // Segments share barWidth minus the separator columns; the last segment
  // absorbs the division remainder so the bar fills barWidth exactly.
  void emitSegmentedBarStdscr(int row, int col, const std::vector<float>& fractions, int numSegments, int barWidth)
  {
    int separators = numSegments - 1;
    int segmentWidth = (barWidth - separators) / numSegments;
    int lastSegmentWidth = (barWidth - separators) - segmentWidth * separators;
    int startCol = col + kLabelWidth + kBracketWidth;

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

  //===================================================================================================================//
  // Emit the trailing per-segment suffix " (0:XX% | 1:XX%)".
  void emitSegmentSuffix(WINDOW* win, const std::vector<float>& fractions)
  {
    ::waddstr(win, buildSegmentSuffix(fractions).c_str());
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

    int bw = barWidthForStdscr(this->x, this->width, numSegments);

    // Draw label and bar on the first row.
    emitLabelStdscr(this->y, this->x, this->barLabel);

    if (numSegments == 1) {
      emitSingleBarStdscr(this->y, this->x + kLabelWidth + kBracketWidth,
                          this->barFractions.empty() ? 0.0f : this->barFractions[0], bw);
    } else {
      emitSegmentedBarStdscr(this->y, this->x, this->barFractions, numSegments, bw);
    }

    // Percentage suffix.
    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), "] %6.3f%%", static_cast<double>(fraction * 100.0f));
    int pctCol = this->x + kLabelWidth + kBracketWidth + bw;
    mvaddstr(this->y, pctCol, pctBuf);

    // Per-segment suffix " (0:XX% | 1:XX%)" right after the percentage, when it fits.
    if (numSegments > 1) {
      std::string suffix = buildSegmentSuffix(this->barFractions);
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

    int bw = barWidthFor(::getmaxx(win), 1);
    emitLabel(win, label);
    emitSingleBar(win, fraction, bw);

    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), "] %6.3f%%", static_cast<double>(fraction * 100.0f));
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
    int bw = barWidthFor(cols, numSegments);

    emitLabel(win, label);
    emitSegmentedBar(win, clamped, numSegments, bw);

    // Compute average percentage across all segments.
    float totalPct = 0.0f;

    for (float f : clamped)
      totalPct += f;

    float avgPct = clamped.empty() ? 0.0f : (totalPct / static_cast<float>(clamped.size())) * 100.0f;

    char pctBuf[16];
    snprintf(pctBuf, sizeof(pctBuf), "] %6.3f%%", static_cast<double>(avgPct));
    ::waddstr(win, pctBuf);

    // Emit the per-segment suffix only when multi-segment and it fits within the window.
    int overhead = kLabelWidth + kBracketWidth + kRightPad + segmentSuffixWidth(numSegments);

    if (numSegments > 1 && segmentSuffixWidth(numSegments) > 0 && bw + overhead <= cols)
      emitSegmentSuffix(win, clamped);

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
