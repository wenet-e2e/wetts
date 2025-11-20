// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//               2025 Binbin Zhang (binbzha@qq.com)
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

#include "model/tts.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "utils/log.h"

#include "frontend/sentence_break.h"
#include "utils/string.h"
#include "utils/utils.h"

namespace wetts {

TTS::TTS(const std::string& encoder_model_path,
         const std::string& decoder_model_path, const std::string& speaker2id,
         const std::string& phone2id, const int sampling_rate,
         std::shared_ptr<TN> tn,
         std::shared_ptr<G2pProsody> g2p_prosody, int chunk_size, int pad_size)
    : vits_(encoder_model_path, decoder_model_path, chunk_size, pad_size),
      tn_(std::move(tn)),
      g2p_prosody_(std::move(g2p_prosody)) {
  sampling_rate_ = sampling_rate;
  ReadTableFile(phone2id, &phone2id_);
  ReadTableFile(speaker2id, &speaker2id_);
}

bool TTS::Text2PhoneIds(const std::string& text,
                        std::vector<int64_t>* phone_ids) {
  phone_ids->clear();
  // 1. TN
  std::string norm_text = tn_->Normalize(text);
  LOG(INFO) << text <<  " --TN--> " << norm_text;
  // 2. G2P: char => pinyin => phones => ids
  std::vector<std::string> phonemes;
  g2p_prosody_->Compute(norm_text, &phonemes);
  // 3. Convert to phone id
  if (phonemes.size() > 0) {
    std::stringstream ss;
    phone_ids->emplace_back(phone2id_["sil"]);
    ss << "sil";
    for (const auto& phone : phonemes) {
      if (phone2id_.count(phone) == 0) {
        LOG(ERROR) << "Can't find `" << phone << "` in phone2id.";
        continue;
      }
      ss << " " << phone;
      phone_ids->emplace_back(phone2id_[phone]);
    }
    LOG(INFO) << "phone sequence " << ss.str();
      return true;
  } else {
    return false;
  }
}

void TTS::Synthesis(const std::string& text, const int sid,
                    std::vector<float>* audio) {
  std::vector<std::string> text_arrs;
  SentenceSegement(text, &text_arrs);
  for (const auto& text : text_arrs) {
    std::vector<int64_t> phonemes;
    bool ok = Text2PhoneIds(text, &phonemes);
    if (ok) {
      std::vector<float> sub_audio;
      vits_.Forward(phonemes, sid, &sub_audio);
      audio->insert(audio->end(), sub_audio.begin(), sub_audio.end());
    }
  }
}

void TTS::SetStreamInfo(const std::string& text, int sid) {
  sid_ = sid;
  text_arrs_.clear();
  cur_text_idx_ = 0;
  new_text_ = true;
  SentenceSegement(text, &text_arrs_);
}

bool TTS::StreamSynthesis(std::vector<float>* audio) {
  if (cur_text_idx_ >= text_arrs_.size()) {
    return true;  // all text done
  }
  if (new_text_) {
    // 连续跳过转换失败的文本片段，直到找到有效的片段
    while (cur_text_idx_ < text_arrs_.size()) {
      std::vector<int64_t> phonemes;
      bool ok = Text2PhoneIds(text_arrs_[cur_text_idx_], &phonemes);
      if (ok && !phonemes.empty()) {
        // 找到有效的片段，设置输入并退出循环
        vits_.SetInput(phonemes, sid_);
        new_text_ = false;
        break;
      }
      // 转换失败或结果为空，跳过当前文本片段，继续下一个
      cur_text_idx_++;
    }
    // 如果所有片段都处理完了，返回 true
    if (cur_text_idx_ >= text_arrs_.size()) {
      return true;
    }
  }
  bool done = vits_.StreamDecode(audio);
  if (done) {
    cur_text_idx_++;
    new_text_ = true;
  }
  return cur_text_idx_ >= text_arrs_.size() ? true : false;
}

int TTS::GetSid(const std::string& name) {
  std::string default_sname = speaker2id_.begin()->first;
  if (speaker2id_.find(name) == speaker2id_.end()) {
    LOG(INFO) << "Invalid speaker name: " << name << ", ";
    LOG(INFO) << "fallback to default speaker: " << default_sname;
    return speaker2id_[default_sname];
  }
  return speaker2id_[name];
}

}  // namespace wetts
