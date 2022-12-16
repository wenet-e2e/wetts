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

namespace wetts {

G2pProsody::G2pProsody(const std::string& g2p_prosody_model,
                       const std::string& phone_file,
                       const std::string& tokenizer_vocab_file,
                       const std::string& lexicon_file)
    : OnnxModel(g2p_prosody_model) {
  // Load phone list file
  std::ifstream is(phone_file);
  std::string line;
  while (getline(is, line)) {
    phones_.emplace_back(line);
  }
  // Load tokenizer
  tokenizer_ = std::make_shared<Tokenizer>(tokenizer_vocab_file);
  lexicon_ = std::make_shared<Lexicon>(lexicon_file);
}

void G2pProsody::Compute(const std::string& str,
                         std::vector<std::string>* phonemes,
                         std::vector<int>* prosody) {
  CHECK(phonemes != nullptr);
  CHECK(prosody != nullptr);
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
  auto phoneme_info = outputs_ort[0].GetTensorTypeAndShapeInfo();
  int phoneme_dim = phoneme_info.GetShape()[2];
  const float* phoneme_data = outputs_ort[0].GetTensorData<float>();
  phonemes->clear();
  // TODO(Binbin Zhang): How to deal with English G2P?
  // Remove [CLS] & [SEQ]
  for (int i = 1; i < num_tokens - 1; i++) {
    std::string phone;
    if (lexicon_->NumProns(tokens[i]) > 1) {
      const float* cur_data = phoneme_data + i * phoneme_dim;
      int best_idx =
          std::max_element(cur_data, cur_data + phoneme_dim) - cur_data;
      phone = phones_[best_idx];
    } else {
      phone = lexicon_->Prons(tokens[i]);
    }
    phonemes->emplace_back(phone);
  }

  auto prosody_info = outputs_ort[1].GetTensorTypeAndShapeInfo();
  int prosody_dim = prosody_info.GetShape()[2];
  const float* prosody_data = outputs_ort[1].GetTensorData<float>();
  prosody->clear();
  // Remove [CLS] & [SEQ]
  for (int i = 1; i < num_tokens - 1; i++) {
    const float* cur_data = prosody_data + i * prosody_dim;
    int best_idx =
        std::max_element(cur_data, cur_data + prosody_dim) - cur_data;
    prosody->emplace_back(best_idx);
  }
}

}  // namespace wetts
