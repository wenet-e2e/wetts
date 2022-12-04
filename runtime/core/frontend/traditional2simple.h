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

#ifndef FRONTEND_TRADITIONAL2SIMPLE_H_
#define FRONTEND_TRADITIONAL2SIMPLE_H_

#include <string>
#include <unordered_map>

namespace wetts {

// Traditional to simplified
// The format of dict_file is like:
// 丟 丢
// 並 并
// 乗 乘
// 乹 乾
// 亁 乾

class Traditional2Simple {
 public:
  explicit Traditional2Simple(const std::string& dict_file);
  std::string Convert(const std::string& in);

 private:
  std::unordered_map<std::string, std::string> t2s_dict_;
};

}  // namespace wetts


#endif  //  FRONTEND_TRADITIONAL2SIMPLE_H_
