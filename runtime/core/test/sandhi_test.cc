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

#include "frontend/sandhi.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using wetts::Sandhi;

TEST(SandhiTest, TwoConsecutiveThirdTone) {
  // 两个连续三声：'xx3' + 'xx3' -> 'xx2' + 'xx3'
  // 例如："你好" ni3 hao3 -> ni2 hao3
  std::string word = "你好";
  std::vector<std::string> pinyin = {"ni3", "hao3"};
  std::vector<std::string> expected = {"ni2", "hao3"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, BuWithFourthTone) {
  // "不" + 四声 => bu2 + 四声
  // 例如："不要" bu4 yao4 -> bu2 yao4
  std::string word = "不要";
  std::vector<std::string> pinyin = {"bu4", "yao4"};
  std::vector<std::string> expected = {"bu2", "yao4"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, BuWithNonFourthTone) {
  // "不" + 非四声 => 不变
  // 例如："不好" bu4 hao3 -> bu4 hao3 (不变)
  std::string word = "不好";
  std::vector<std::string> pinyin = {"bu4", "hao3"};
  std::vector<std::string> expected = {"bu4", "hao3"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, YiWithFourthTone) {
  // "一" + 四声 => yi2 + 四声
  // 例如："一个" yi1 ge4 -> yi2 ge4
  std::string word = "一个";
  std::vector<std::string> pinyin = {"yi1", "ge4"};
  std::vector<std::string> expected = {"yi2", "ge4"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, YiWithNonFourthTone) {
  // "一" + 非四声 => 不变
  // 例如："一起" yi1 qi3 -> yi1 qi3 (不变)
  std::string word = "一起";
  std::vector<std::string> pinyin = {"yi1", "qi3"};
  std::vector<std::string> expected = {"yi4", "qi3"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, ComplexCase) {
  // 复杂情况：三声变调 + "不"/"一"变调
  // 例如："不很好" bu4 hen3 hao3 -> bu4 hen2 hao3 (只有三声变调)
  std::string word = "不很好";
  std::vector<std::string> pinyin = {"bu4", "hen3", "hao3"};
  std::vector<std::string> expected = {"bu4", "hen2", "hao3"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, NoSandhiCase) {
  // 不需要变调的情况
  // 例如："很好" 如果第二个字不是三声，则不变
  std::string word = "很好";
  std::vector<std::string> pinyin = {"hen3", "hao1"};
  std::vector<std::string> expected = {"hen3", "hao1"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}

TEST(SandhiTest, SingleCharacter) {
  // 单字情况：不需要变调（因为没有下一个字）
  std::string word = "好";
  std::vector<std::string> pinyin = {"hao3"};
  std::vector<std::string> expected = {"hao3"};
  Sandhi(word, &pinyin);
  ASSERT_EQ(pinyin, expected);
}
