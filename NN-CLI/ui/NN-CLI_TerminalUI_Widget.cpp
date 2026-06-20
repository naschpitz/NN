#include "NN-CLI_TerminalUI_Widget.hpp"

namespace NN_CLI
{

  //===================================================================================================================//
  //-- Core interface --//
  //===================================================================================================================//

  void TerminalUI_Widget::resize(int width, int height, int x, int y)
  {
    this->width = width;
    this->height = height;
    this->x = x;
    this->y = y;
  }

  //===================================================================================================================//
  //-- Core interface --//
  //===================================================================================================================//

  bool TerminalUI_Widget::handleEvent(int /*ch*/)
  {
    return false;
  }

  //===================================================================================================================//
  //-- Child management --//
  //===================================================================================================================//

  void TerminalUI_Widget::addChild(std::unique_ptr<TerminalUI_Widget> child)
  {
    if (child) {
      this->children.push_back(std::move(child));
      this->markDirty();
    }
  }

  //===================================================================================================================//

  std::unique_ptr<TerminalUI_Widget> TerminalUI_Widget::removeChild(int index)
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    auto removed = std::move(this->children[index]);
    this->children.erase(this->children.begin() + index);
    this->markDirty();

    return removed;
  }

  //===================================================================================================================//

  TerminalUI_Widget* TerminalUI_Widget::getChild(int index) const
  {
    if (index < 0 || index >= static_cast<int>(this->children.size()))
      return nullptr;

    return this->children[index].get();
  }

} // namespace NN_CLI
