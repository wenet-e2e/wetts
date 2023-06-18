// Copyright (c) 2023 Zhendong Peng (pzd17@tsinghua.org.cn)
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

#include "utils/utils.h"

namespace wetts {

void ReadTableFile(const std::string& file,
                   std::unordered_map<std::string, int>* map) {
  std::fstream infile(file);
  std::string left;
  int right;
  while (infile >> left >> right) {
    (*map)[left] = right;
  }
}

void ReadTableFile(const std::string& file,
                   std::unordered_map<std::string, std::string>* map) {
  std::ifstream infile(file);
  std::string line;
  while (getline(infile, line)) {
    int pos = line.find_first_of(" \t", 0);
    std::string key = line.substr(0, pos);
    std::string value = line.substr(pos + 1, line.size() - pos);
    (*map)[key] = value;
  }
}

void ReadTableFile(
    const std::string& file,
    std::unordered_map<std::string, std::vector<std::string>>* map) {
  std::ifstream infile(file);
  std::string line;
  while (getline(infile, line)) {
    std::vector<std::string> strs;
    SplitString(line, &strs);
    CHECK_GE(strs.size(), 2);
    std::string key = strs[0];
    strs.erase(strs.begin());
    (*map)[key] = strs;
  }
}

}  // namespace wetts
