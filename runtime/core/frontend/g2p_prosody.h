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

#ifndef FRONTEND_G2P_PROSODY_H_
#define FRONTEND_G2P_PROSODY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "frontend/g2p_en.h"
#include "frontend/lexicon.h"
#include "frontend/word_break.h"
#include "model/onnx_model.h"

namespace wetts {

// Unified G2P & Prosody model
class G2pProsody {
 public:
  explicit G2pProsody(const std::string& g2p_prosody_model,
                      const std::string& vocab, const std::string& lexicon_file,
                      const std::string& pinyin2id,
                      const std::string& pinyin2phones,
                      std::shared_ptr<G2pEn> g2p_en = nullptr);
  void Tokenize(const std::vector<std::string>& words,
                std::vector<int64_t>* token_ids,
                std::vector<int>* token_offsets);
  void Compute(const std::string& str, std::vector<std::string>* phonemes);
  void Forward(const std::vector<std::string>& words,
               const std::vector<int64_t>& token_ids,
               const std::vector<int>& token_offsets,
               std::vector<std::string>* pinyins,
               std::vector<std::string>* prosodys);

 private:
  const std::string CLS_ = "[CLS]";
  const std::string SEP_ = "[SEP]";
  const std::string UNK_ = "[UNK]";
  OnnxModel model_;
  std::unordered_map<std::string, int> vocab_;
  std::unordered_map<std::string, int> phones_;
  std::shared_ptr<G2pEn> g2p_en_;
  std::shared_ptr<Lexicon> lexicon_;
  std::shared_ptr<WordBreak> word_break_;
  std::unordered_map<std::string, std::vector<std::string>> pinyin2phones_;
};

}  // namespace wetts

#endif  // FRONTEND_G2P_PROSODY_H_
