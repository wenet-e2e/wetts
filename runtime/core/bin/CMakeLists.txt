add_executable(tts_main tts_main.cc)
target_link_libraries(tts_main PUBLIC gflags tts)

if(BUILD_SERVER)
  add_executable(http_server_main http_server_main.cc)
  target_link_libraries(http_server_main PUBLIC gflags http_server tts jsoncpp_lib)
endif()
