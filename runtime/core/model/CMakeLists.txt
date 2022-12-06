add_library(tts_model STATIC onnx_tts_model.cc)
target_link_libraries(tts_model PUBLIC utils)
