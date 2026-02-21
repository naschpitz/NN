cmake_minimum_required(VERSION 3.14)

project(NN-CLI LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Concurrent)

# Add CNN as subdirectory — it brings ANN and OpenCLWrapper transitively via PUBLIC
if(NOT TARGET CNN)
  add_subdirectory(extern/CNN)
endif()

add_executable(NN-CLI
  main.cpp
  NN-CLI_Loader.cpp
  NN-CLI_ProgressBar.cpp
  NN-CLI_Runner.cpp
  NN-CLI_Utils.cpp
)

# Link to CNN — ANN, OpenCLWrapper, Qt::Core, OpenCL are propagated automatically via PUBLIC
target_link_libraries(NN-CLI
    PRIVATE CNN
    PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent
)

include(GNUInstallDirs)
install(TARGETS NN-CLI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
