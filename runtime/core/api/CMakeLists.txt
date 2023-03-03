if(ONNX)
 add_library(wetts_api SHARED wetts_api.cc)
 target_link_libraries(wetts_api PUBLIC tts_model)
endif()
