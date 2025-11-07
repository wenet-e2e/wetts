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

#ifdef BUILD_WITH_FST
#include "fst/fstlib.h"
#endif

namespace wetts {

#ifdef BUILD_WITH_FST
using fst::StdVectorFst;
using fst::ProjectType::PROJECT_OUTPUT;
using fst::StringTokenType::BYTE;

using ArcIterator = fst::ArcIterator<fst::StdVectorFst>;
using StringPrinter = fst::StringPrinter<fst::StdArc>;
using StateIterator = fst::StateIterator<fst::StdVectorFst>;
using StringCompiler = fst::StringCompiler<fst::StdArc>;

StdVectorFst ShortestPath(const std::string& input, const StdVectorFst* fst);

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::string* output);

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::vector<int>* olabels);
#endif

class G2pEn {
 public:
  G2pEn(const std::string& cmudict, const std::string& model,
        const std::string& sym);

  void Convert(const std::string& grapheme, std::vector<std::string>* phonemes);
  std::string Convert(const std::string& grapheme);

 private:
  std::unordered_map<std::string, std::vector<std::string>> cmudict_;
#ifdef BUILD_WITH_FST
  std::shared_ptr<fst::StdVectorFst> model_;
  std::shared_ptr<fst::SymbolTable> sym_;
#endif
};

}  // namespace wetts

#endif  // FRONTEND_G2P_EN_H_
