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

#include "utils/log.h"

#include "frontend/fst.h"
#include "utils/string.h"
#include "utils/utils.h"

namespace wetts {

G2pEn::G2pEn(const std::string& cmudict, const std::string& model,
             const std::string& sym) {
  ReadTableFile(cmudict, &cmudict_);
  model_.reset(fst::StdVectorFst::Read(model));
  sym_.reset(fst::SymbolTable::ReadText(sym));
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
  }
}

std::string G2pEn::Convert(const std::string& grapheme) {
  std::vector<std::string> phonemes;
  Convert(grapheme, &phonemes);
  return JoinString(" ", phonemes);
}

}  // namespace wetts
