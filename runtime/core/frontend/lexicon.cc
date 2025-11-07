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

#include "frontend/lexicon.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/log.h"

#include "utils/string.h"

namespace wetts {

const char Lexicon::UNK[] = "<UNK>";

Lexicon::Lexicon(const std::string& lexicon_file) {
  std::ifstream is(lexicon_file);
  std::string line;
  while (getline(is, line)) {
    size_t pos = line.find(' ');
    CHECK(pos != std::string::npos);
    std::string word = line.substr(0, pos);
    std::string prons_str = line.substr(pos + 1);
    std::vector<std::string> prons;
    SplitStringToVector(prons_str, ",", true, &prons);
    lexicon_[word] = std::move(prons);
  }
  unk_.emplace_back(UNK);
}

int Lexicon::NumProns(const std::string& word) {
  if (lexicon_.find(word) != lexicon_.end()) {
    return lexicon_[word].size();
  } else {
    return 0;
  }
}

const std::vector<std::string>& Lexicon::Prons(const std::string& word) {
  if (lexicon_.find(word) != lexicon_.end()) {
    return lexicon_[word];
  } else {
    return unk_;
  }
}

}  // namespace wetts
