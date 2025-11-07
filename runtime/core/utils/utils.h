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

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/string.h"

namespace wetts {

void ReadTableFile(const std::string& file,
                   std::unordered_map<std::string, int>* map);

void ReadTableFile(const std::string& file,
                   std::unordered_map<std::string, std::string>* map);

void ReadTableFile(
    const std::string& file,
    std::unordered_map<std::string, std::vector<std::string>>* map);

}  // namespace wetts

#endif  // UTILS_UTILS_H_
