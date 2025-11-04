// Copyright (c) 2025 GPT(binbzha@qq.com)
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

#include <cctype>
#include <string>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

#include "utils/string.h"

namespace wetts {

static bool IsSentenceDelimiterChar(const std::string& ch) {
  // 支持英文与中文（全角）常见句子/短语分隔符以及换行
  static const std::unordered_set<std::string> kDelims = {
      ".", ";", "!", "?", "。", "；", "！", "？", "\n", "\r"};
  return kDelims.find(ch) != kDelims.end();
}

void Segement(const std::string& text, std::vector<std::string>* sentences,
              size_t max_clause_len) {
  if (sentences == nullptr) {
    return;
  }
  sentences->clear();
  if (text.empty()) {
    return;
  }

  std::string current;
  current.reserve(text.size());
  size_t current_chars = 0;
  size_t last_safe_index = 0;  // current 中最近的安全切割点（字节下标）
  bool in_ascii_word = false;

  std::vector<std::string> chars;
  SplitUTF8StringToChars(text, &chars);
  for (const auto& ch : chars) {
    if (IsSentenceDelimiterChar(ch)) {
      std::string trimmed = Trim(current);
      if (!trimmed.empty()) {
        sentences->emplace_back(trimmed);
      }
      current.clear();
      current_chars = 0;
      last_safe_index = 0;
      in_ascii_word = false;
      continue;
    }
    // 记录安全切割点（不切英文单词）
    bool is_ascii_alnum =
        (ch.size() == 1) && std::isalnum(static_cast<unsigned char>(ch[0]));
    bool is_space = (ch == " " || ch == "\t");
    if (is_space) {
      last_safe_index = current.size();
      in_ascii_word = false;
    } else if (!in_ascii_word && is_ascii_alnum) {
      last_safe_index = current.size();
      in_ascii_word = true;
    } else if (in_ascii_word && !is_ascii_alnum) {
      last_safe_index = current.size();
      in_ascii_word = false;
    }

    current += ch;
    ++current_chars;

    if (max_clause_len > 0 && current_chars >= max_clause_len) {
      if (last_safe_index > 0) {
        std::string left = Trim(current.substr(0, last_safe_index));
        if (!left.empty()) {
          sentences->emplace_back(left);
        }
        std::string right = current.substr(last_safe_index);
        current.swap(right);
        current_chars = UTF8StringLength(current);
        last_safe_index = 0;
        in_ascii_word = false;
      } else {
        // 没有安全点（例如超长英文单词），直接强制切分
        std::string left = Trim(current);
        if (!left.empty()) {
          sentences->emplace_back(left);
        }
        current.clear();
        current_chars = 0;
        in_ascii_word = false;
      }
    }
  }

  std::string trimmed = Trim(current);
  if (!trimmed.empty()) {
    sentences->emplace_back(trimmed);
  }
}

}  // namespace wetts
