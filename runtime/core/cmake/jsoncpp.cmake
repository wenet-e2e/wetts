FetchContent_Declare(jsoncpp
  URL      https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.9.3.zip
  URL_HASH SHA256=7853fe085ddd5da94b9795f4b520689c21f2753c4a8f7a5097410ee6136bf671
)
FetchContent_MakeAvailable(jsoncpp)
include_directories(${jsoncpp_SOURCE_DIR}/include)
