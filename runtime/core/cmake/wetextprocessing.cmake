FetchContent_Declare(wetextprocessing

  URL https://github.com/wenet-e2e/WeTextProcessing/archive/refs/tags/0.1.0.tar.gz
  URL_HASH SHA256=5edb3a37230e8764cc0d9f4551246776745fd51ef004b3d584bfa68076ea514a
  SOURCE_SUBDIR runtime
)
FetchContent_MakeAvailable(wetextprocessing)
include_directories(${openfst_SOURCE_DIR}/src/include)
include_directories(${wetextprocessing_SOURCE_DIR}/runtime)
link_directories(${wetextprocessing_BINARY_DIR})
