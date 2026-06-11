#ifndef NN_CLI_TERMINALUI_TABLE_HPP
#define NN_CLI_TERMINALUI_TABLE_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"

#include <functional>
#include <string>
#include <vector>

namespace NN_CLI
{

  // Reusable ASCII table formatter and terminal widget that produces
  // std::vector<std::string> of formatted lines and can render them directly
  // onto stdscr at the widget's (x, y) position.
  //
  // Inherits from TerminalUI_Widget so it can participate in the MVC view
  // hierarchy.  The unified draw() override renders the table at the widget's
  // position; the render() method returns formatted lines for callers that
  // need raw string output (e.g. Panel content lines).
  //
  // Column widths are computed at render time from the maxWidth and per-column
  // width hints, following the same algorithm as TerminalUI::rebuildEpochLines():
  //   1. Subtract structural overhead (borders, padding) from maxWidth.
  //   2. Distribute the remaining data width proportionally among columns
  //      based on their widthHint values.
  //
  // Supports three primary use cases:
  //   - Key-value two-column tables (SummaryTable style, with centered title).
  //   - Three-column profiler tables (TrainingProfiler style, header + data).
  //   - Multi-column epoch tables (rebuildEpochLines style, 5+ columns).

  class TerminalUI_Table : public TerminalUI_Widget
  {
    public:
      //-- Types --//

      enum class Align { LEFT, RIGHT, CENTER };

      struct Column {
          std::string name;
          int widthHint = 0;
          Align align = Align::LEFT;
          // Optional per-cell formatter applied before alignment/padding.
          std::function<std::string(const std::string&)> formatter;
      };

      using Row = std::vector<std::string>;

      //-- Ctors --//

      TerminalUI_Table();
      explicit TerminalUI_Table(int maxWidth);
      TerminalUI_Table(std::vector<Column> columns, int maxWidth);

      //-- Configuration --//

      void setColumns(std::vector<Column> columns);
      void setTitle(const std::string& title);
      void setMaxWidth(int maxWidth);

      //-- Data --//

      void addRow(const Row& cells);
      void addRows(const std::vector<Row>& rows);
      void clearRows();

      //-- Widget overrides --//

      // Render the table onto stdscr at (x, y), drawing up to height rows.
      // The widget's width constrains the effective maxWidth.
      void draw() override;

      // Update position and size.  Syncs maxWidth with the new width.
      void resize(int width, int height, int x, int y) override;

      //-- Rendering --//

      // Build and return the complete table as formatted lines.
      // When a title is set, it replaces the header row (SummaryTable pattern).
      // When no title is set, the column-name header row is rendered.
      // An empty data set produces one placeholder row.
      std::vector<std::string> render() const;

      //-- Accessors --//

      int getMaxWidth() const;
      int columnCount() const;

      // Triggers width computation if dirty; returns the resolved column widths.
      const std::vector<int>& getComputedWidths() const;

      // A standalone separator line matching the current column widths.
      // Useful for callers who need to insert section breaks between renders.
      std::string separator() const;

    private:
      //-- Methods --//
      void computeWidths() const;
      std::string formatCell(const std::string& text, int width, Align align) const;
      std::string makeRow(const Row& cells) const;
      std::string makeSeparator() const;
      std::string makeTitleRow() const;

      //-- Members --//
      std::vector<Column> columns;
      std::string title;
      std::vector<Row> rows;
      int maxWidth = 80;
      mutable std::vector<int> computedWidths;
      mutable bool widthsDirty = true;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_TABLE_HPP
