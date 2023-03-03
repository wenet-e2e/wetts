cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(wetts VERSION 0.1)

option(ONNX "whether to build with ONNX" ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base-${CMAKE_CXX_COMPILER_ID}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")

include(glog)
include(gflags)
if(ONNX)
  include(onnx)
endif()
include(wetextprocessing)
include(pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(model)
add_subdirectory(frontend)

# wetts api
pybind11_add_module(_wetts cpp/binding.cc)
target_link_libraries(_wetts PRIVATE wetts_api)
