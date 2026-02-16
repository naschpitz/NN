cmake_minimum_required(VERSION 3.14)

project(ANN-CLI LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

add_executable(ANN-CLI
  main.cpp
)

# Add include directories for dependencies
set(LIB_ANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/ANN")
include_directories(${LIB_ANN_DIR})

set(LIB_NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann")
include_directories(${LIB_NLOHMANN_DIR})

set(LIB_OPENCL_WRAPPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/OpenCLWrapper")
include_directories(${LIB_OPENCL_WRAPPER_DIR})

# OpenCL library
find_package(OpenCL REQUIRED)

# Link libraries in correct order (dependents first, then their dependencies)
target_link_libraries(ANN-CLI
    "${LIB_ANN_DIR}/libANN.a"
    "${LIB_OPENCL_WRAPPER_DIR}/libOpenCLWrapper.a"
    Qt${QT_VERSION_MAJOR}::Core
    OpenCL::OpenCL
)

include(GNUInstallDirs)
install(TARGETS ANN-CLI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
