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

#include <iostream>
#include <fstream>
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
  std::string space_str = AddSpaceForChineseChar(str);
  std::vector<std::string> split_strs;
  SplitString(space_str, &split_strs);

  tokens->clear();
  tokens->emplace_back(cls_token_);
  for (const std::string& token : split_strs) {
    if (IsChineseChar(token)) {
      // TODO(Binbin Zhang): Chinese OOV
      tokens->emplace_back(token);
    } else {
      // TODO(Binbin Zhang): Normal to lower case? Or we already done it in TN
      // Word piece Encoding, greedy match
      for (int start = 0; start < token.size();) {
        bool oov = true;
        for (int end = token.size(); end > start; end--) {
          std::string substr = token.substr(start, end - start);
          if (start > 0) {
            substr = "##" + substr;
          }
          if (vocab_.find(substr) != vocab_.end()) {
            tokens->emplace_back(substr);
            start = end;
            oov = false;
            break;
          }
        }
        if (oov) {
          tokens->emplace_back("[UNK]");
          start++;
        }
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
