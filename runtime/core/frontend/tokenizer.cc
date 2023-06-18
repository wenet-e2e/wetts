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

#include "frontend/tokenizer.h"

#include <fstream>
#include <iostream>
#include <string>

#include "utils/string.h"

namespace wetts {

Tokenizer::Tokenizer(const std::string& vocab_file) {
  std::ifstream is(vocab_file);
  std::string line;
  int id = 0;
  while (getline(is, line)) {
    vocab_[line] = id;
    id++;
  }
}

void Tokenizer::Tokenize(const std::string& str,
                         std::vector<std::string>* tokens,
                         std::vector<int64_t>* token_ids) const {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(str, &chars);

  tokens->emplace_back(cls_token_);
  for (int i = 0; i < chars.size(); ++i) {
    std::string& token = chars[i];
    if (token == " ") {
      continue;
    }
    if (IsChineseChar(token)) {
      tokens->emplace_back(token);
    } else if (IsAlpha(token)) {
      tokens->emplace_back(token);
      while (i + 1 < chars.size() && IsAlpha(chars[i + 1])) {
        ++i;
        tokens->back() += chars[i];
      }
    }
  }
  tokens->emplace_back(sep_token_);

  // Get Id
  token_ids->clear();
  token_ids->resize(tokens->size());
  // Already process OOV before, so just use it here.
  for (int i = 0; i < tokens->size(); i++) {
    (*token_ids)[i] = vocab_.at((*tokens)[i]);
  }
}

}  // namespace wetts
