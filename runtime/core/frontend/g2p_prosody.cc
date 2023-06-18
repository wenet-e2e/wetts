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

#include "frontend/g2p_prosody.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "utils/string.h"
#include "utils/utils.h"

namespace wetts {

G2pProsody::G2pProsody(const std::string& g2p_prosody_model,
                       const std::string& vocab, const std::string& char2pinyin,
                       const std::string& pinyin2id,
                       const std::string& pinyin2phones)
    : OnnxModel(g2p_prosody_model) {
  // Load tokenizer
  tokenizer_ = std::make_shared<Tokenizer>(vocab);
  lexicon_ = std::make_shared<Lexicon>(char2pinyin);

  // Load phone list file
  std::ifstream is(pinyin2id);
  std::string line;
  int idx = 0;
  while (getline(is, line)) {
    phones_[line] = idx;
    idx++;
  }
  ReadTableFile(pinyin2phones, &pinyin2phones_);
}

void G2pProsody::Compute(const std::string& str,
                         std::vector<std::string>* phonemes) {
  CHECK(phonemes != nullptr);
  std::vector<int64_t> token_ids;
  std::vector<std::string> tokens;
  tokenizer_->Tokenize(str, &tokens, &token_ids);
  int num_tokens = token_ids.size();
  const int64_t inputs_shape[] = {1, num_tokens};
  auto inputs_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info_, token_ids.data(), num_tokens, inputs_shape, 2);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  auto outputs_ort = session_->Run(
      Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
      ort_inputs.size(), output_node_names_.data(), 2);
  auto pinyin_info = outputs_ort[0].GetTensorTypeAndShapeInfo();
  int pinyin_dim = pinyin_info.GetShape()[2];
  const float* pinyin_data = outputs_ort[0].GetTensorData<float>();

  auto prosody_info = outputs_ort[1].GetTensorTypeAndShapeInfo();
  int prosody_dim = prosody_info.GetShape()[2];
  const float* prosody_data = outputs_ort[1].GetTensorData<float>();

  std::vector<std::string> pinyins;
  std::vector<std::string> prosodys;
  // TODO(Binbin Zhang): How to deal with English G2P?
  // Remove [CLS] & [SEP]
  for (int t = 1; t < num_tokens - 1; t++) {
    std::string pinyin;
    if (lexicon_->NumProns(tokens[t]) > 1) {
      const std::vector<std::string>& possible_prons =
          lexicon_->Prons(tokens[t]);
      std::vector<float> possible_vals;
      for (const std::string& pron : possible_prons) {
        int pron_offset = phones_[pron];
        possible_vals.emplace_back(
            *(pinyin_data + t * pinyin_dim + pron_offset));
      }
      int best_idx =
          std::max_element(possible_vals.begin(), possible_vals.end()) -
          possible_vals.begin();
      pinyin = possible_prons[best_idx];
    } else {
      pinyin = lexicon_->Prons(tokens[t])[0];
      // The token could be an english word or punctuation
      if (pinyin == Lexicon::UNK) {
        pinyin = tokens[t];
      }
    }
    pinyins.emplace_back(pinyin);

    const float* cur_data = prosody_data + t * prosody_dim;
    int best_idx =
        std::max_element(cur_data, cur_data + prosody_dim) - cur_data;
    prosodys.emplace_back("#" + std::to_string(best_idx));
  }

  for (int idx = 0; idx < pinyins.size(); idx++) {
    std::string pinyin = pinyins[idx];
    std::string prosody = prosodys[idx];
    transform(pinyin.begin(), pinyin.end(), pinyin.begin(), ::tolower);
    if (pinyin2phones_.count(pinyin) > 0) {
      std::vector<std::string>& phones = pinyin2phones_[pinyin];
      phonemes->insert(phonemes->end(), phones.begin(), phones.end());
      phonemes->emplace_back(prosody);
    } else {
      LOG(ERROR) << "Pinyin " << pinyin << " not found in pinyin2phones";
    }
  }
}

}  // namespace wetts
