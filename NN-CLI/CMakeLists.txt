cmake_minimum_required(VERSION 3.14)

project(ANN-CLI LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Concurrent)

# Add dependencies as subdirectories (order matters: OpenCLWrapper before ANN)
add_subdirectory(extern/OpenCLWrapper)
add_subdirectory(extern/ANN)

add_executable(ANN-CLI
  main.cpp
  ANN-CLI_Utils.cpp
  ANN-CLI_Loader.cpp
)

# Link only to ANN - OpenCLWrapper, Qt::Core, OpenCL are propagated automatically via PUBLIC
target_link_libraries(ANN-CLI
    PRIVATE ANN
    PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent
)

include(GNUInstallDirs)
install(TARGETS ANN-CLI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
