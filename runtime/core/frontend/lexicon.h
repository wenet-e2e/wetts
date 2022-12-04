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

#ifndef FRONTEND_LEXICON_H_
#define FRONTEND_LEXICON_H_

#include <string>
#include <unordered_map>
#include <utility>

namespace wetts {

// Lexicon, the format is like
// 今 jin1
// 天 tian1
// 好 hao3,hao4
// TODAY T AH D EY
// DESERT D EH1 Z ER0 T,D IH0 Z ER1 T

class Lexicon {
 public:
  explicit Lexicon(const std::string& lexicon_file);
  int NumProns(const std::string& word);
  std::string Prons(const std::string& word);

 private:
  std::unordered_map<std::string, std::pair<int, std::string>> lexicon_;
};

}  // namespace wetts

#endif  // FRONTEND_LEXICON_H_
