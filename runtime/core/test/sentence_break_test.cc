// Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
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

#include "frontend/sentence_break.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using wetts::SentenceSegement;

TEST(SentenceBreakTest, ChinesePunctuations) {
  std::string text = "我爱编程，但是今天下雨了。明天呢？好吧！";
  std::vector<std::string> sentences;
  SentenceSegement(text, &sentences);
  // 逗号不进行切分，标点符号包含在返回的句子中
  std::vector<std::string> expect = {"我爱编程，但是今天下雨了。", "明天呢？",
                                     "好吧！"};
  ASSERT_EQ(sentences, expect);
}

TEST(SentenceBreakTest, EnglishPunctuations) {
  std::string text = "Hello, world! Are you OK? Yes; good.";
  std::vector<std::string> sentences;
  SentenceSegement(text, &sentences);
  // 逗号不进行切分，标点符号包含在返回的句子中
  std::vector<std::string> expect = {"Hello, world!", "Are you OK?", "Yes;",
                                     "good."};
  ASSERT_EQ(sentences, expect);
}

TEST(SentenceBreakTest, MaxLengthSplit) {
  // 无标点，依赖最大长度强制切分 + 不切英文单词
  std::string text = "abc def ghi jkl";
  std::vector<std::string> sentences;
  SentenceSegement(text, &sentences, 4);
  std::vector<std::string> expect = {"abc", "def", "ghi", "jkl"};
  ASSERT_EQ(sentences, expect);
}

TEST(SentenceBreakTest, ChineseMaxLengthSplit) {
  // 中文无空格，达到最大长度时允许强制切分（不涉及英文单词）
  std::string text = "我爱编程学习";  // 6 个中文字符
  std::vector<std::string> sentences;
  SentenceSegement(text, &sentences, 3);
  std::vector<std::string> expect = {"我爱编", "程学习"};
  ASSERT_EQ(sentences, expect);
}

TEST(SentenceBreakTest, CommaAsSafeBreakPoint) {
  // 测试逗号作为安全切割点，且不切分连续数字
  std::string text = "11月10日，第十五届全国运动会武术套路比赛在广州南沙体育馆收官。来自广州的\"00后\"志愿者李镁雪也结束了她的\"最后一班岗\"";  // NOLINT
  std::vector<std::string> sentences;
  SentenceSegement(text, &sentences, 32);
  // 逗号是安全切割点，当超过最大长度时可以在逗号处切分
  // 句号是句子分隔符，会直接切分
  // "00后"是连续数字，不应该被切分
  std::vector<std::string> expect = {
    "11月10日，",
    "第十五届全国运动会武术套路比赛在广州南沙体育馆收官。",
    "来自广州的\"00后\"志愿者李镁雪也结束了她的\"最后一班岗\"",
  };
  ASSERT_EQ(sentences, expect);
}
