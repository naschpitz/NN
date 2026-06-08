cmake_minimum_required(VERSION 3.14)

project(NN-Server LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Network Concurrent)

# Add CNN as subdirectory — it brings ANN and OpenCLWrapper transitively via PUBLIC
if(NOT TARGET CNN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../CNN ${CMAKE_CURRENT_BINARY_DIR}/CNN)
endif()

add_executable(NN-Server
  main.cpp
  NN-Server_CorePool.cpp
  NN-Server_HttpServer.cpp
  NN-Server_ImageLoader.cpp
  NN-Server_Loader.cpp
  NN-Server_Logger.cpp
  NN-Server_RequestHandler.cpp
)

# Header-only libraries
target_include_directories(NN-Server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
)

# Link to CNN — ANN, OpenCLWrapper, Qt::Core, OpenCL are propagated automatically via PUBLIC
target_link_libraries(NN-Server
    PRIVATE CNN
    PRIVATE Qt6::Network
    PRIVATE Qt6::Concurrent
)

include(GNUInstallDirs)
install(TARGETS NN-Server
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Test executable — integration tests that start the server and send HTTP requests
add_executable(test_endpoints
  tests/test_main.cpp
  tests/test_endpoints.cpp
)
target_include_directories(test_endpoints PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
)
target_link_libraries(test_endpoints PRIVATE
    Qt6::Core
    Qt6::Network
)

# Logger unit tests — tests Logger directly without starting the server
add_executable(test_logger
  tests/test_logger_main.cpp
  tests/test_logger.cpp
  NN-Server_Logger.cpp
)
target_include_directories(test_logger PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_link_libraries(test_logger PRIVATE
    Qt6::Core
    Qt6::Network
)

