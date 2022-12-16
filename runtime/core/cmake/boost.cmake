FetchContent_Declare(boost
  URL https://boostorg.jfrog.io/artifactory/main/beta/1.81.0.beta1/source/boost_1_81_0_b1.tar.gz
  URL_HASH SHA256=135f03965b50d05baae45f49e4b7f2f3c545ff956b4500342f8fb328b8207a90
)
FetchContent_MakeAvailable(boost)
include_directories(${boost_SOURCE_DIR})

if(MSVC)
  add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
endif()
