add_library(onnx_model STATIC onnx_model.cc)
target_link_libraries(onnx_model PUBLIC onnxruntime utils)

add_library(tts_model STATIC tts_model.cc)
target_link_libraries(tts_model PUBLIC wetext_processor frontend)
