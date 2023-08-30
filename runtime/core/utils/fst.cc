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

#include "utils/fst.h"

#include "fst/rmepsilon.h"

namespace wetts {

StdVectorFst ShortestPath(const std::string& input, const StdVectorFst* fst) {
  StdVectorFst input_fst;
  static StringCompiler compiler(BYTE);
  compiler(input, &input_fst);

  StdVectorFst lattice;
  fst::Compose(input_fst, *fst, &lattice);
  StdVectorFst shortest_path;
  fst::ShortestPath(lattice, &shortest_path, 1, true);
  return shortest_path;
}

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::string* output) {
  StdVectorFst lattice = ShortestPath(input, fst);
  static StringPrinter printer(BYTE);
  printer(lattice, output);
}

void ShortestPath(const std::string& input, const StdVectorFst* fst,
                  std::vector<int>* olabels) {
  StdVectorFst lattice = ShortestPath(input, fst);
  fst::Project(&lattice, PROJECT_OUTPUT);
  fst::RmEpsilon(&lattice);
  fst::TopSort(&lattice);

  for (StateIterator siter(lattice); !siter.Done(); siter.Next()) {
    ArcIterator aiter(lattice, siter.Value());
    if (!aiter.Done()) {
      olabels->emplace_back(aiter.Value().olabel);
    }
  }
}

}  // namespace wetts
