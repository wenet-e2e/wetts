FetchContent_Declare(wetextprocessing
  URL https://github.com/wenet-e2e/WeTextProcessing/archive/refs/tags/0.1.3.tar.gz
  URL_HASH SHA256=2f1c81649b2f725a5825345356be9dccb9699965cf44c9f0e842f5c0d4b6ba61
  SOURCE_SUBDIR runtime
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${openfst_SOURCE_DIR}/src/include)
include_directories(${wetextprocessing_SOURCE_DIR}/runtime)
link_directories(${wetextprocessing_BINARY_DIR})
