cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wetts VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(ONNX ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base-${CMAKE_CXX_COMPILER_ID}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(glog)
include(gflags)
include(onnx)
include(wetextprocessing)

add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(model)
add_subdirectory(bin)
