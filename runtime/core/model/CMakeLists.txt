add_library(onnx_model STATIC onnx_model.cc)
target_link_libraries(onnx_model PUBLIC onnxruntime utils)

add_library(vits_model STATIC vits_model.cc)
target_link_libraries(vits_model PUBLIC onnx_model)

add_library(tts STATIC tts.cc)
if(FST)
target_link_libraries(tts PUBLIC wetext_processor frontend vits_model)
else()
target_link_libraries(tts PUBLIC frontend vits_model)
endif()
