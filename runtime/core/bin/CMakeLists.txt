set(FRONTEND_BINS
  frontend_traditional2simple_main
  frontend_lexicon_main
  frontend_tokenizer_main
  frontend_g2p_prosody_main
)

foreach(name IN LISTS FRONTEND_BINS)
  add_executable(${name} ${name}.cc)
  target_link_libraries(${name} PUBLIC frontend gflags)
endforeach()

add_executable(tts_main tts_main.cc)
target_link_libraries(tts_main PUBLIC gflags_nothreads_static tts_model)
