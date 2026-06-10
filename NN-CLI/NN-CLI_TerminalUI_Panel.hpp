#ifndef NN_CLI_TERMINALUI_PANEL_HPP
#define NN_CLI_TERMINALUI_PANEL_HPP

#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Panel container that owns bounds and handles framing/content drawing for a
  // specific area of the screen.  Unifies the left-column panels (Configuration,
  // Epochs) and the right-column Timing panel under one drawing/scrolling model.
  //
  // Drawing is performed directly on stdscr; the panel stores its (y, x, h, w)
  // bounds and the caller positions it via resize().  The three draw methods
  // (drawFrame, drawContent, drawScrollbar) are independent so callers can
  // compose them — e.g. drawing all frames first, then all content.
  //
  // Color pair convention (matches TerminalUI init):
  //   2 = CYAN   (inactive / normal)
  //   3 = YELLOW (active / highlighted)

  class TerminalUI_Panel
  {
    public:
      //-- Types --//

      // Per-panel scroll state.  When autoScroll is true the view is pinned to
      // the newest line until the user scrolls manually (applyScrollInput clears
      // autoScroll on any recognized key).
      struct ScrollState {
          int offset = 0;
          bool autoScroll = false;
      };

      //-- Ctors --//

      TerminalUI_Panel();
      TerminalUI_Panel(int y, int x, int h, int w, std::string title, int colorPair);

      //-- Layout --//

      // Update bounds.  Does not redraw — call drawFrame/drawContent/drawScrollbar after.
      void resize(int y, int x, int h, int w);

      // Update the panel title (used by drawFrame).
      void setTitle(const std::string& title);

      // Update the color pair for the title (2=CYAN inactive, 3=YELLOW active).
      void setColorPair(int colorPair);

      //-- Content --//

      // Replace the content lines (copied; does not trim or reset scroll).
      void setLines(const std::vector<std::string>& lines);

      // Enable or disable auto-scroll mode.  When enabled, scrollOffset() returns
      // the maximum scroll position so the view is always pinned to the bottom.
      void setAutoScroll(bool autoScroll);

      //-- Drawing --//

      // Draw the ACS border frame and colored title.
      void drawFrame() const;

      // Draw visible content lines inside the frame, clipped to contentWidth().
      void drawContent() const;

      // Draw the vertical scrollbar on the right edge (no-op if content fits).
      void drawScrollbar() const;

      //-- Input --//

      // Handle a keypress for scrolling (Up/Down/k/j, PgUp/PgDn, Home/End).
      // Returns true if `ch` was a recognized scroll key.  Clears autoScroll.
      bool applyScrollInput(int ch);

      //-- Accessors --//

      // Effective content width, accounting for the scrollbar column when needed.
      // Returns w-4 when content fits, w-5 when the scrollbar is shown.
      int contentWidth() const;

      int y() const
      {
        return this->y_;
      }

      int x() const
      {
        return this->x_;
      }

      int h() const
      {
        return this->h_;
      }

      int w() const
      {
        return this->w_;
      }

      const std::string& title() const
      {
        return this->title_;
      }

      int colorPair() const
      {
        return this->colorPair_;
      }

      ScrollState& scrollState()
      {
        return this->scroll_;
      }

      const ScrollState& scrollState() const
      {
        return this->scroll_;
      }

      const std::vector<std::string>& lines() const
      {
        return this->lines_;
      }

    private:
      //-- Methods --//

      // Number of rows available for content inside the frame (h - 2).
      int contentHeight() const;

      // Resolved scroll offset: autoScroll returns max, otherwise clamped offset.
      int scrollOffset() const;

      //-- Members --//

      int y_ = 0;
      int x_ = 0;
      int h_ = 0;
      int w_ = 0;
      std::string title_;
      int colorPair_ = 2; // CYAN (inactive) by default
      ScrollState scroll_;
      std::vector<std::string> lines_;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PANEL_HPP
