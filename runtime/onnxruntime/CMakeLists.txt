cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(wetts VERSION 0.1)

set(CMAKE_VERBOSE_MAKEFILE OFF)
option(BUILD_TESTING "whether to build unit test" OFF)
option(BUILD_SERVER "whether to build server binaries" OFF)
option(ONNX "whether to build with ONNX" ON)
option(FST "wether to build with FST" ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base-${CMAKE_CXX_COMPILER_ID}" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${fc_base})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include(glog)
include(gflags)
include(onnxruntime)
if(FST)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -DFST")
  include(wetextprocessing)
endif()
if(BUILD_TESTING)
  include(gtest)
endif()

add_subdirectory(utils)
add_subdirectory(model)
add_subdirectory(frontend)

if(BUILD_SERVER)
  include(boost)
  include(jsoncpp)
  add_subdirectory(http)
endif()

add_subdirectory(bin)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()
