FetchContent_Declare(wetextprocessing

  URL https://github.com/wenet-e2e/WeTextProcessing/archive/refs/tags/0.1.0.tar.gz
  URL_HASH SHA256=072193d8d2ba396759b13da2f7cc4da7a80045419103479b8cf5d2d2194d215c
  SOURCE_SUBDIR runtime
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${openfst_SOURCE_DIR}/src/include)
include_directories(${wetextprocessing_SOURCE_DIR}/runtime)
link_directories(${wetextprocessing_BINARY_DIR})
