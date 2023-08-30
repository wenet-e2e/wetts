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

#ifndef FRONTEND_G2P_EN_H_
#define FRONTEND_G2P_EN_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fst/fstlib.h"

namespace wetts {

class G2pEn {
 public:
  G2pEn(const std::string& cmudict, const std::string& model,
        const std::string& sym);

  void Convert(const std::string& grapheme, std::vector<std::string>* phonemes);

 private:
  std::unordered_map<std::string, std::vector<std::string>> cmudict_;
  std::shared_ptr<fst::StdVectorFst> model_;
  std::shared_ptr<fst::SymbolTable> sym_;
};

}  // namespace wetts

#endif  // FRONTEND_G2P_EN_H_
