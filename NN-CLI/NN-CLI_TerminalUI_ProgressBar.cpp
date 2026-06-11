#include "NN-CLI_TerminalUI_ProgressBar.hpp"

#include <algorithm>
#include <curses.h>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
  //-- Layout constants shared by every ncurses progress bar so they all line up
  //-- vertically in the Training panel. --//
  constexpr int kLabelWidth    = 28;  // labels are left-padded to this width
  constexpr int kBracketWidth  =  2;  // the " [" written right after the label
  constexpr int kRightPad      = 20;  // reserved on the right for "] 100.000%" suffix (+ slack)
  constexpr int kSegmentInfoPerSegment = 9;  // width of one " | N:XXX%" cell in the per-segment suffix

  //-- Color pairs (configured in TerminalUI::init). --//
  constexpr int kBarColor = 1;  // green filled bar segment

  //===================================================================================================================//
  // Width reserved on the right for the per-segment "(0:XX% | 1:XX%)" suffix (0 when single-segment).
  int segmentSuffixWidth(int segments)
  {
    return segments > 1 ? kSegmentInfoPerSegment * segments - 1 : 0;
  }

  //===================================================================================================================//
  // Usable bar width inside `win`, mirrored across all bars so their brackets line up.
  int barWidthFor(WINDOW* win, int segments)
  {
    int cols = ::getmaxx(win);
    return std::max(10, cols - kLabelWidth - kBracketWidth - kRightPad - segmentSuffixWidth(segments));
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
  // Emit a single solid bar: `filled` green blocks, then light shading out to `barWidth`.
  void emitSingleBar(WINDOW* win, float fraction, int barWidth)
  {
    int filled = std::clamp(static_cast<int>(fraction * barWidth), 0, barWidth);

    ::wattron(win, COLOR_PAIR(kBarColor));

    for (int i = 0; i < filled; i++)
      ::waddstr(win, "█");

    ::wattroff(win, COLOR_PAIR(kBarColor));

    for (int i = filled; i < barWidth; i++)
      ::waddstr(win, "░");
  }

  //===================================================================================================================//
  // Emit a segmented bar, one equal-width segment separated by "│".
  void emitSegmentedBar(WINDOW* win, const std::vector<float>& fractions, int numSegments, int barWidth)
  {
    int segmentWidth = barWidth / numSegments;

    for (int seg = 0; seg < numSegments; seg++) {
      float pct = (seg < static_cast<int>(fractions.size())) ? fractions[seg] : 0.0f;
      emitSingleBar(win, pct, segmentWidth);

      if (seg < numSegments - 1)
        ::waddstr(win, "│");
    }
  }

  //===================================================================================================================//
  // Emit the trailing per-segment suffix " (0:XX% | 1:XX%)".
  void emitSegmentSuffix(WINDOW* win, const std::vector<float>& fractions)
  {
    std::ostringstream info;
    info << " (";

    for (size_t s = 0; s < fractions.size(); s++) {
      info << s << ":" << std::setw(3) << static_cast<int>(fractions[s] * 100) << "%";

      if (s + 1 < fractions.size())
        info << " | ";
    }

    info << ")";
    ::waddstr(win, info.str().c_str());
  }

} // namespace

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Ctors --//
  //===================================================================================================================//

  TerminalUI_ProgressBar::TerminalUI_ProgressBar()
  {
  }

  //===================================================================================================================//
  //-- ncurses Rendering --//
  //===================================================================================================================//

  void TerminalUI_ProgressBar::renderSingleBar(WINDOW* win, const std::string& label, float fraction)
  {
    if (!win)
      return;

    fraction = std::clamp(fraction, 0.0f, 1.0f);

    ::werase(win);

    int bw = barWidthFor(win, 1);
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
    int bw = barWidthFor(win, numSegments);

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
      out << (i < filledWidth ? "█" : "░");

    out << "] " << current << "/" << total << "  " << std::fixed << std::setprecision(1) << (percent * 100.0f) << "%";
    out << "   ";

    std::cout << out.str() << std::flush;

    if (current == total)
      std::cout << std::endl;
  }

} // namespace NN_CLI
