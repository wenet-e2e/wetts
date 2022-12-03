add_executable(tts_main tts_main.cc)
target_link_libraries(tts_main PUBLIC gflags_nothreads_static utils tts_model)
