add_library(utils STATIC
  string.cc
  onnx_utils.cc
)

target_link_libraries(utils PUBLIC onnxruntime glog)
