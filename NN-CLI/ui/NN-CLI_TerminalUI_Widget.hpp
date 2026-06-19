#ifndef NN_CLI_TERMINALUI_WIDGET_HPP
#define NN_CLI_TERMINALUI_WIDGET_HPP

namespace NN_CLI
{

  // Abstract base class for all terminal UI elements in the MVC view hierarchy.
  // Every widget owns a position (x, y) and size (width, height) within the
  // terminal screen.  The resize() method updates these bounds and propagates
  // to child widgets in container classes; draw() renders the widget onto
  // stdscr at its current position; handleEvent() provides an optional hook
  // for user input.
  //
  // Ownership model: parent containers (Window, Panel) hold children via
  // std::unique_ptr<TerminalUI_Widget>.  Destroying a parent destroys all its
  // children.

  class TerminalUI_Widget
  {
    public:
      //-- Ctors / Dtors --//

      TerminalUI_Widget() = default;

      virtual ~TerminalUI_Widget() = default;

      TerminalUI_Widget(const TerminalUI_Widget&) = delete;
      TerminalUI_Widget& operator=(const TerminalUI_Widget&) = delete;
      TerminalUI_Widget(TerminalUI_Widget&&) = delete;
      TerminalUI_Widget& operator=(TerminalUI_Widget&&) = delete;

      //-- Core interface --//

      // Render this widget onto the terminal at its current (x, y) position.
      virtual void draw() = 0;

      // Update position and size.  Container widgets propagate to children.
      //   width  — horizontal extent (columns)
      //   height — vertical extent (rows)
      //   x      — column offset from the left edge of the screen
      //   y      — row offset from the top edge of the screen
      virtual void resize(int width, int height, int x, int y);

      // Handle a keypress event.  Returns true if the event was consumed.
      virtual bool handleEvent(int ch);

      //-- Dirty tracking --//

      // True if this widget (or any descendant) has changed since the last
      // draw() cleared the flag.  The UI thread checks this to decide whether
      // a repaint is needed, so unchanged widgets incur zero rendering cost.
      // All widget mutation happens under the window's UI mutex, so the flag
      // itself needs no atomic.
      virtual bool isDirtyTree() const
      {
        return this->dirty;
      }

      //-- Accessors --//

      int getX() const
      {
        return this->x;
      }

      int getY() const
      {
        return this->y;
      }

      int getWidth() const
      {
        return this->width;
      }

      int getHeight() const
      {
        return this->height;
      }

    protected:
      //-- Dirty tracking --//

      // Called by setters that mutate visible widget state (text, rows, bar
      // fraction, child list, ...).  Implemented here so every widget marks
      // itself dirty the same way without repeating the line.
      void markDirty()
      {
        this->dirty = true;
      }

      // Called by a widget's own draw() after it has finished painting itself
      // (and, for containers, its children).
      void clearDirty()
      {
        this->dirty = false;
      }

      //-- Members --//

      int x = 0;
      int y = 0;
      int width = 0;
      int height = 0;

      bool dirty = true; // true initially so the first paint occurs
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_WIDGET_HPP
