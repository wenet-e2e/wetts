// Copyright (c) 2025, Binbin Zhang(binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "frontend/word_break.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_set>
#include <vector>

using wetts::WordBreak;

static WordBreak MakeWB() {
  std::unordered_set<std::string> dict = {"北京",     "大学", "北京大学",
                                          "中国",     "人民", "银行",
                                          "人民银行", "办理", "业务"};
  return WordBreak(dict);
}

TEST(WordBreakTest, BasicMM) {
  auto wb = MakeWB();
  std::vector<std::string> words;
  wb.Segment("我爱北京大学", &words);
  // 最大前向匹配应优先匹配 "北京大学"
  std::vector<std::string> expect = {"我", "爱", "北京大学"};
  ASSERT_EQ(words, expect);
}

TEST(WordBreakTest, EnglishDigitGrouping) {
  auto wb = MakeWB();
  std::vector<std::string> words;
  wb.Segment("abc123 xyz456", &words);
  std::vector<std::string> expect = {"abc123", " ", "xyz456"};
  ASSERT_EQ(words, expect);
}

TEST(WordBreakTest, MixedCnEn) {
  auto wb = MakeWB();
  std::vector<std::string> words;
  wb.Segment("去Bank123办理业务", &words);
  // 非字典部分的英文数字连续片段应整体返回
  std::vector<std::string> expect = {"去", "Bank123", "办理", "业务"};
  ASSERT_EQ(words, expect);
}

TEST(WordBreakTest, DigitsOnly) {
  auto wb = MakeWB();
  std::vector<std::string> words;
  wb.Segment("2025年11月3日", &words);
  // 连续数字整体返回
  // 注意："年"、"月"、"日"不在词典，按字节返回
  std::vector<std::string> expect = {"2025", "年", "11", "月", "3", "日"};
  ASSERT_EQ(words, expect);
}
