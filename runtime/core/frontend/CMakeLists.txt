add_library(frontend STATIC
  lexicon.cc
  g2p_en.cc
  g2p_prosody.cc
  word_break.cc
  sentence_break.cc
  sandhi.cc
)

target_link_libraries(frontend PUBLIC onnx_model)
