add_library(frontend STATIC
  t2s.cc
  lexicon.cc
  tokenizer.cc
)

target_link_libraries(frontend PUBLIC utils glog)
