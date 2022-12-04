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

#include "frontend/traditional2simple.h"

#include <glog/logging.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/string.h"

namespace wetts {

Traditional2Simple::Traditional2Simple(const std::string& dict_file) {
  std::ifstream is(dict_file);
  std::string line;
  while (getline(is, line)) {
    std::vector<std::string> strs;
    SplitString(line, &strs);
    CHECK_GE(strs.size(), 2);
    t2s_dict_[strs[0]] = strs[1];
  }
}

std::string Traditional2Simple::Convert(const std::string& in) {
  std::vector<std::string> chars;
  std::string out;
  SplitUTF8StringToChars(in, &chars);
  for (const auto& x : chars) {
    if (t2s_dict_.find(x) != t2s_dict_.end()) {
      out += t2s_dict_[x];
    } else {
      out += x;
    }
  }
  return out;
}

}  // namespace wetts

