add_library(frontend STATIC
  lexicon.cc
  tokenizer.cc
  g2p_prosody.cc
)

target_link_libraries(frontend PUBLIC onnx_model)
