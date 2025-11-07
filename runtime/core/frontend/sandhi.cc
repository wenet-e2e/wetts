// Copyright (c) 2022, Binbin Zhang (binbzha@qq.com)
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

#include <string>
#include <vector>

#include "utils/log.h"

#include "utils/string.h"

namespace wetts {

// Please refer:
// https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/tone_sandhi.py
// // NO_LINT
void Sandhi(const std::string& word, std::vector<std::string>* pinyin) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(word, &chars);
  CHECK_EQ(chars.size(), pinyin->size());
  for (int i = 0; i < chars.size() - 1; i++) {
    int cur_tone = (*pinyin)[i].back() - '0';
    int next_tone = (*pinyin)[i + 1].back() - '0';
    // 'xx3' + 'xx3' -> 'xx2' ++ 'xx3'
    if (cur_tone == 3 && next_tone == 3) {
      (*pinyin)[i].back() = '2';
    }
    // 不, 不 + xx4 => bu2 + xx4， eg 不要(bu2 yao)
    if (chars[i] == "不" && next_tone == 4) {
      (*pinyin)[i].back() = '2';
    }
    // 一
    if (chars[i] == "一") {
      if (i > 0 && chars[i - 1] == "第") {
        (*pinyin)[i].back() = '1';
      } else if (next_tone == 4) {
        //  一 + xx4 => yi2 + xx4, eg 一个(yi2 ge4)
        (*pinyin)[i].back() = '2';
      } else {
        //  一 + xxn => yi4 + xxn, eg 一起(yi4 qi3)
        (*pinyin)[i].back() = '4';
      }
    }
  }
}

}  // namespace wetts
