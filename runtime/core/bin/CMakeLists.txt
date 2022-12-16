set(FRONTEND_BINS
  frontend_lexicon_main
  frontend_tokenizer_main
  frontend_g2p_prosody_main
)

foreach(name IN LISTS FRONTEND_BINS)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PUBLIC frontend gflags)
endforeach()

add_executable(tts_main tts_main.cc)
target_link_libraries(tts_main PUBLIC gflags tts_model)

add_executable(http_server_main http_server_main.cc)
target_link_libraries(http_server_main PUBLIC gflags http_server tts_model)
