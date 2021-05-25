cmake_minimum_required(VERSION 3.18)
project(albert CXX)

set(ALBERT_SANITIZE "" CACHE STRING "-fsanitize compiler options")
if (ALBERT_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=${ALBERT_SANITIZE}")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=${ALBERT_SANITIZE}")
endif (ALBERT_SANITIZE)

# -----------------------------------------------------------------------------
# External project dependencies
# -----------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(fmt
  GIT_REPOSITORY     https://github.com/fmtlib/fmt.git
  GIT_TAG            master)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(ce
  GIT_REPOSITORY     git@github.com:ldalessa/ce.git
  GIT_TAG            main)
set(CE_BUILD_TESTING OFF)
FetchContent_MakeAvailable(ce)

add_library(albert_lib INTERFACE)
target_include_directories(albert_lib INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(albert_lib INTERFACE ce::ce fmt::fmt)
target_compile_features(albert_lib INTERFACE cxx_std_20)
add_library(albert::albert ALIAS albert_lib)

add_subdirectory(tests)
