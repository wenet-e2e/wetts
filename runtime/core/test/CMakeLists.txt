enable_testing()

add_executable(word_break_test
  word_break_test.cc
)

target_link_libraries(word_break_test
  PRIVATE
    frontend
    gtest
    gtest_main
)

add_test(NAME word_break_test COMMAND word_break_test)
