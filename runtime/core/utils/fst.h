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

#ifndef UTILS_FST_H_
#define UTILS_FST_H_

#include <string>
#include <vector>

#include "fst/fstlib.h"
#include "utils/log.h"

#include "utils/string.h"

using fst::StdVectorFst;
using fst::ProjectType::PROJECT_OUTPUT;
using fst::StringTokenType::BYTE;

using ArcIterator = fst::ArcIterator<fst::StdVectorFst>;
using StringPrinter = fst::StringPrinter<fst::StdArc>;
using StateIterator = fst::StateIterator<fst::StdVectorFst>;
using StringCompiler = fst::StringCompiler<fst::StdArc>;

namespace wetts {

StdVectorFst ShortestPath(const std::string& input, const StdVectorFst* fst);

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::string* output);

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::vector<int>* olabels);

}  // namespace wetts

#endif  // UTILS_FST_H_
