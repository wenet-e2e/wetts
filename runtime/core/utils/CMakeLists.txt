add_library(utils STATIC
  fst.cc
  string.cc
  utils.cc
)

target_link_libraries(utils PUBLIC glog fst)
