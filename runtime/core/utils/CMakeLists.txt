add_library(utils STATIC
  fst.cc
  string.cc
  utils.cc
)

if(ANDROID)
    find_library(alog log)  # andorid log
    target_link_libraries(utils PUBLIC ${alog})
else()
    # For non-Android platforms, link glog
    target_link_libraries(utils PUBLIC glog)
endif()
