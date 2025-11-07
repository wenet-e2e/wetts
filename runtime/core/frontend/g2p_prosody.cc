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

#include "utils/log.h"

#include "utils/string.h"
#include "utils/utils.h"

#include "frontend/word_break.h"
#include "frontend/sandhi.h"

namespace wetts {

G2pProsody::G2pProsody(const std::string& g2p_prosody_model,
                       const std::string& g2p_prosody_vocab,
                       const std::string& lexicon_file,
                       const std::string& pinyin2id,
                       const std::string& pinyin2phones,
                       std::shared_ptr<G2pEn> g2p_en)
    : g2p_en_(std::move(g2p_en)), model_(g2p_prosody_model) {
  std::ifstream in(g2p_prosody_vocab);
  std::string line;
  int id = 0;
  while (getline(in, line)) {
    g2p_vocab_[line] = id;
    id++;
  }
  lexicon_ = std::make_shared<Lexicon>(lexicon_file);
  word_break_ = std::make_shared<WordBreak>(lexicon_file);

  // Load phone list file
  std::ifstream is(pinyin2id);
  int idx = 0;
  while (getline(is, line)) {
    phones_[line] = idx;
    idx++;
  }
  ReadTableFile(pinyin2phones, &pinyin2phones_);
}

void G2pProsody::Tokenize(const std::vector<std::string>& words,
                          std::vector<int64_t>* token_ids,
                          std::vector<int>* token_offsets) {
  token_ids->clear();
  token_ids->emplace_back(g2p_vocab_.at(CLS_));
  token_offsets->clear();
  int offset = 1;  // 0 is taken by CLS_
  for (const std::string& word : words) {
    token_offsets->emplace_back(offset);
    if (lexicon_->NumProns(word) > 0) {
      // Split word into single chinese chars
      std::vector<std::string> chars;
      SplitUTF8StringToChars(word, &chars);
      for (const std::string& ch : chars) {
        token_ids->emplace_back(g2p_vocab_.at(ch));
        offset++;
      }
    } else if (word[0] < 128 && std::isalnum(word[0])) {
      // English or digit word, Convert english word to UNK
      token_ids->emplace_back(g2p_vocab_.at(UNK_));
      offset++;
    } else {
      std::string v = g2p_vocab_.find(word) != g2p_vocab_.end() ? word : UNK_;
      token_ids->emplace_back(g2p_vocab_.at(v));
      offset++;
    }
  }
  token_ids->emplace_back(g2p_vocab_.at(SEP_));
}

void G2pProsody::Forward(const std::vector<std::string>& words,
                         const std::vector<int64_t>& token_ids,
                         const std::vector<int>& token_offsets,
                         std::vector<std::string>* pinyins,
                         std::vector<std::string>* prosodys) {
  pinyins->clear();
  prosodys->clear();
  int num_tokens = token_ids.size();
  const int64_t inputs_shape[] = {1, num_tokens};
  auto inputs_ort = Ort::Value::CreateTensor<int64_t>(
      model_.memory_info(), const_cast<int64_t*>(token_ids.data()), num_tokens,
      inputs_shape, 2);
  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  auto outputs_ort = model_.Run(ort_inputs);
  auto pinyin_info = outputs_ort[0].GetTensorTypeAndShapeInfo();
  int pinyin_dim = pinyin_info.GetShape()[2];
  const float* pinyin_data = outputs_ort[0].GetTensorData<float>();
  auto prosody_info = outputs_ort[1].GetTensorTypeAndShapeInfo();
  int prosody_dim = prosody_info.GetShape()[2];
  const float* prosody_data = outputs_ort[1].GetTensorData<float>();
  // TODO(Binbin Zhang): How to deal with English G2P?
  // Remove [CLS] & [SEP]
  int num_words = words.size();
  for (int i = 0; i < num_words; i++) {
    const std::string& word = words[i];
    int num_chars = UTF8StringLength(word);
    int offset = token_offsets[i];
    int prosody_offset = offset;
    std::string pinyin;
    std::string prosody;
    if (lexicon_->NumProns(word) == 0) {
      // 0. OOV or English
      pinyins->emplace_back(word);
    } else if (lexicon_->NumProns(word) == 1) {
      // 1. Word or non polyphone char
      pinyin = lexicon_->Prons(word)[0];
      pinyins->emplace_back(pinyin);
      // We assume it's #0 inside the word
      for (int j = 0; j < num_chars - 1; j++) {
        prosody += "#0 ";
        prosody_offset++;
      }
    } else {  // lexicon_->NumProns(word) > 1
      // 2. Single chinese polyphone char, g2p
      CHECK_EQ(num_chars, 1);
      const std::vector<std::string>& possible_prons = lexicon_->Prons(word);
      std::vector<float> possible_vals;
      for (const std::string& pron : possible_prons) {
        int pron_offset = phones_[pron];
        possible_vals.emplace_back(
            *(pinyin_data + offset * pinyin_dim + pron_offset));
      }
      int best_idx =
          std::max_element(possible_vals.begin(), possible_vals.end()) -
          possible_vals.begin();
      pinyin = possible_prons[best_idx];
      pinyins->emplace_back(pinyin);
    }
    // Compute prosody in the word boundary
    const float* cur_data = prosody_data + prosody_offset * prosody_dim;
    int best_idx =
        std::max_element(cur_data, cur_data + prosody_dim) - cur_data;
    prosody += ("#" + std::to_string(best_idx));
    prosodys->emplace_back(prosody);
  }
}

void G2pProsody::Compute(const std::string& str,
                         std::vector<std::string>* phonemes) {
  CHECK(phonemes != nullptr);
  phonemes->clear();
  // 1. First segment the input text into words
  std::vector<std::string> words;
  word_break_->Segment(str, &words);
  // 2. Chinese g2p & prosody
  std::vector<int64_t> token_ids;
  std::vector<int> token_offsets;
  Tokenize(words, &token_ids, &token_offsets);
  std::vector<std::string> pinyins;
  std::vector<std::string> prosodys;
  Forward(words, token_ids, token_offsets, &pinyins, &prosodys);
  // 3. English words(lookup or phonetisaurus g2p for English OOV)
  for (int i = 0; i < words.size(); i++) {
    if (CheckEnglishWord(words[i])) {
      std::vector<std::string> phones;
      pinyins[i] = g2p_en_->Convert(ToLower(words[i]));
    }
  }
  // 4. Concat phoneme+prosody to final sequence
  for (int idx = 0; idx < words.size(); idx++) {
    const std::string& word = words[idx];
    std::vector<std::string> pinyin, prosody;
    SplitString(pinyins[idx], &pinyin);
    SplitString(prosodys[idx], &prosody);
    if (lexicon_->NumProns(word) > 0) {
      CHECK_EQ(pinyin.size(), prosody.size());
      Sandhi(word, &pinyin);
      for (int n = 0; n < pinyin.size(); n++) {
        if (pinyin2phones_.count(pinyin[n]) > 0) {
          std::vector<std::string>& phones = pinyin2phones_[pinyin[n]];
          phonemes->insert(phonemes->end(), phones.begin(), phones.end());
          phonemes->emplace_back(prosody[n]);
        } else {
          LOG(ERROR) << "Pinyin " << pinyin[n] << " not found in pinyin2phones";
        }
      }
    } else if (CheckEnglishWord(word)) {
      CHECK_EQ(prosody.size(), 1);
      phonemes->insert(phonemes->end(), pinyin.begin(), pinyin.end());
      phonemes->emplace_back(prosody[0]);
    } else {
      // Not English, Not in Lexicon, ignore now
      // TODO(Binbin Zhang): Deal punct
      LOG(WARNING) << "Ignore word " << word;
    }
    VLOG(2) << "Word, g2p & prosody: " << word << " "
            << pinyins[idx] << " " << prosodys[idx];
  }
  // Last token should be "#4"
  phonemes->back() = "#4";
}


}  // namespace wetts
