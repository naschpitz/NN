cmake_minimum_required(VERSION 3.14)

project(ANN LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

add_library(ANN STATIC
  ANN_Core.hpp ANN_Core.cpp
  ANN_ActvFunc.hpp ANN_ActvFunc.cpp
  ANN_Utils.hpp ANN_Utils.cpp
  ANN_CoreMode.hpp ANN_CoreMode.cpp
  ANN_CoreCPU.hpp ANN_CoreCPU.cpp
  ANN_CoreGPU.hpp ANN_CoreGPU.cpp
  ANN_CoreType.hpp ANN_CoreType.cpp
  ANN_LayersConfig.hpp ANN_LayersConfig.cpp
  Kernels.cpp.cl
  IdxHelper.cpp.cl
  ActvFunc.hpp.cl ActvFunc.cpp.cl
  Defines.hpp..cl
  LICENSE.md
)

target_link_libraries(ANN PRIVATE Qt${QT_VERSION_MAJOR}::Core)

target_compile_definitions(ANN PRIVATE ANN_LIBRARY)

#Begin add the OpenCLWrapper library
set(LIB_OPENCL_WRAPPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/OpenCLWrapper")

include_directories(${LIB_OPENCL_WRAPPER_DIR})
target_link_libraries(ANN PRIVATE "${LIB_OPENCL_WRAPPER_DIR}/libOpenCLWrapper.a")
#End add the OpenCLWrapper library

#Begin add the nlohmann library
set(LIB_NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann")

include_directories(${LIB_NLOHMANN_DIR})
#End add the nlohmann library
