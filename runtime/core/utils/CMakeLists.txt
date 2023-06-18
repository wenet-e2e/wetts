add_library(utils STATIC string.cc utils.cc)

target_link_libraries(utils PUBLIC glog)
