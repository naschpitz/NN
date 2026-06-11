#ifndef NN_CLI_TERMINALUI_PANEL_HPP
#define NN_CLI_TERMINALUI_PANEL_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"

#include <memory>
#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Container widget that draws a bordered panel with optional title, manages
  // scrollable text content, and owns child widgets laid out within its
  // content area.
  //
  // Inherits from TerminalUI_Widget so it can participate in the MVC view
  // hierarchy alongside Window, ProgressBar, Table, and other widgets.
  //
  // Drawing is performed directly on stdscr via the ncurses mvaddch/mvaddstr
  // family.  The three legacy draw methods (drawFrame, drawContent,
  // drawScrollbar) are kept for backward compatibility; the unified draw()
  // override invokes all three and then renders children.
  //
  // Resize propagation: resize() updates this panel's bounds and then calls
  // layoutChildren() to reposition child widgets within the content area.
  //
  // Color pair convention (matches TerminalUI init):
  //   2 = CYAN   (inactive / normal)
  //   3 = YELLOW (active / highlighted)

  class TerminalUI_Panel : public TerminalUI_Widget
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

      // Primary constructor: title and color pair.  Position/size are set via
      // resize() or the backward-compatible constructor below.
      TerminalUI_Panel(const std::string& title, int colorPair);

      // Backward-compatible constructor: (row, col, rows, cols, title, colorPair).
      // Maps directly to the Widget's position and size fields.
      TerminalUI_Panel(int y, int x, int h, int w, std::string title, int colorPair);

      //-- Layout --//

      // Widget override: update bounds and propagate to children via
      // layoutChildren().
      //
      // Parameter order: (width, height, x, y) — width before position.
      void resize(int width, int height, int x, int y) override;

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

      // Unified Widget override: draws frame, content, scrollbar, and children.
      void draw() override;

      // Draw the ACS border frame and colored title.
      void drawFrame() const;

      // Draw visible content lines inside the frame, clipped to contentWidth().
      void drawContent() const;

      // Draw the vertical scrollbar on the right edge (no-op if content fits).
      void drawScrollbar() const;

      //-- Input --//

      // Widget override: delegates to applyScrollInput() for scroll keys, then
      // propagates to children.
      bool handleEvent(int ch) override;

      // Handle a keypress for scrolling (Up/Down/k/j, PgUp/PgDn, Home/End).
      // Returns true if `ch` was a recognized scroll key.  Clears autoScroll.
      bool applyScrollInput(int ch);

      //-- Child management --//

      // Take ownership of a child widget positioned within this panel.
      void addChild(std::unique_ptr<TerminalUI_Widget> child);

      // Release and return the child at the given index, or nullptr if out of range.
      std::unique_ptr<TerminalUI_Widget> removeChild(int index);

      int childCount() const
      {
        return static_cast<int>(this->children.size());
      }

      TerminalUI_Widget* getChild(int index) const;

      //-- Layout (children) --//

      // Reposition child widgets within this panel's content area.  The default
      // implementation gives every child the full content area; override for
      // custom strategies.
      virtual void layoutChildren();

      //-- Accessors --//

      // Effective content width, accounting for the scrollbar column when needed.
      // Returns width-4 when content fits, width-5 when the scrollbar is shown.
      int contentWidth() const;

      // Height available for content lines inside the frame (height - 2).
      int contentHeight() const;

      // Backward-compatible aliases that map to the Widget base accessors.
      int getY() const
      {
        return this->y;
      }

      int getX() const
      {
        return this->x;
      }

      int getH() const
      {
        return this->height;
      }

      int getW() const
      {
        return this->width;
      }

      const std::string& getTitle() const
      {
        return this->title;
      }

      int getColorPair() const
      {
        return this->colorPair;
      }

      ScrollState& scrollState()
      {
        return this->scroll;
      }

      const ScrollState& scrollState() const
      {
        return this->scroll;
      }

      const std::vector<std::string>& getLines() const
      {
        return this->lines;
      }

    protected:
      //-- Methods --//

      // Resolved scroll offset: autoScroll returns max, otherwise clamped offset.
      int scrollOffset() const;

      //-- Members --//

      std::string title;
      int colorPair = 2; // CYAN (inactive) by default
      ScrollState scroll;
      std::vector<std::string> lines;
      std::vector<std::unique_ptr<TerminalUI_Widget>> children;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PANEL_HPP
