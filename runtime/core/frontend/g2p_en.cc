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

#include "frontend/g2p_en.h"

#include <string>
#include <vector>

#ifdef BUILD_WITH_FST
#include "fst/rmepsilon.h"
#endif

#include "utils/log.h"

#include "utils/string.h"
#include "utils/utils.h"

namespace wetts {

#ifdef BUILD_WITH_FST
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
#endif

G2pEn::G2pEn(const std::string& cmudict, const std::string& model,
             const std::string& sym) {
  ReadTableFile(cmudict, &cmudict_);
#ifdef BUILD_WITH_FST
  model_.reset(fst::StdVectorFst::Read(model));
  sym_.reset(fst::SymbolTable::ReadText(sym));
#endif
}

void G2pEn::Convert(const std::string& grapheme,
                    std::vector<std::string>* phonemes) {
  if (cmudict_.count(grapheme) > 0) {
    *phonemes = cmudict_[grapheme];
  } else if (grapheme.size() < 4) {
    // Speak short oov letter by letter, such as `ASR` and `TTS`
    for (int i = 0; i < grapheme.size(); i++) {
      std::string token{grapheme[i]};
      std::vector<std::string>& phones = cmudict_[token];
      phonemes->insert(phonemes->end(), phones.begin(), phones.end());
      if (i < grapheme.size() - 1) {
        // TODO(zhendong.peng): use prosody dict instead of hard code
        phonemes->emplace_back("#0");
      }
    }
  } else {
#ifdef BUILD_WITH_FST
    std::vector<std::string> graphemes;
    SplitStringToVector(grapheme, "-", true, &graphemes);
    for (int i = 0; i < graphemes.size(); ++i) {
      std::vector<int> olabels;
      ShortestPath(graphemes[i], model_.get(), &olabels);
      for (auto olabel : olabels) {
        const auto& phoneme = sym_->Find(olabel);
        phonemes->emplace_back(phoneme);
      }
      if (i != graphemes.size() - 1) {
        phonemes->emplace_back("#0");
      }
    }
#endif
  }
}

std::string G2pEn::Convert(const std::string& grapheme) {
  std::vector<std::string> phonemes;
  Convert(grapheme, &phonemes);
  return JoinString(" ", phonemes);
}

}  // namespace wetts
