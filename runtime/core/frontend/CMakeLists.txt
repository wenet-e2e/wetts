add_library(frontend STATIC
  traditional2simple.cc
  lexicon.cc
  tokenizer.cc
  g2p_prosody.cc
)

target_link_libraries(frontend PUBLIC utils)
