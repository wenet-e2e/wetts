cmake_minimum_required(VERSION 3.4.1)
project(wetts CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE on)

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

file(GLOB ONNXRUNTIME_INCLUDE_DIRS ${build_DIR}/onnxruntime*.aar/headers)
file(GLOB ONNXRUNTIME_LINK_DIRS ${build_DIR}/onnxruntime*.aar/jni/${ANDROID_ABI})
link_directories(${ONNXRUNTIME_LINK_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

set(openfst_BINARY_DIR ${build_DIR}/wenet-openfst-android-1.0.2.aar/jni)
link_directories(${openfst_BINARY_DIR}/${ANDROID_ABI})
link_libraries(log gflags_nothreads glog fst)
include_directories(${openfst_BINARY_DIR}/include)

include(wetextprocessing)
include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(frontend)
add_subdirectory(model)
add_dependencies(frontend wetextprocessing)

add_library(wetts SHARED wetts.cc)
target_link_libraries(wetts PUBLIC tts_model onnxruntime)
