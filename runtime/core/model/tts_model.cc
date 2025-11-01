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

#include "model/tts_model.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "utils/string.h"
#include "utils/utils.h"

namespace wetts {

TtsModel::TtsModel(const std::string& encoder_model_path,
                   const std::string& decoder_model_path,
                   const std::string& speaker2id, const std::string& phone2id,
                   const int sampling_rate,
                   std::shared_ptr<wetext::Processor> tn,
                   std::shared_ptr<G2pProsody> g2p_prosody, int chunk_size,
                   int pad_size)
    : encoder_(encoder_model_path),
      decoder_(decoder_model_path),
      tn_(std::move(tn)),
      g2p_prosody_(std::move(g2p_prosody)),
      s_chunk_size_(chunk_size),
      s_pad_size_(pad_size) {
  sampling_rate_ = sampling_rate;
  ReadTableFile(phone2id, &phone2id_);
  ReadTableFile(speaker2id, &speaker2id_);
}

std::vector<Ort::Value> TtsModel::ForwardEncoder(
    const std::vector<int64_t>& phonemes, const int sid) {
  int num_phones = phonemes.size();
  const int64_t inputs_shape[] = {1, num_phones};
  auto inputs_ort = Ort::Value::CreateTensor<int64_t>(
      encoder_.memory_info(), const_cast<int64_t*>(phonemes.data()), num_phones,
      inputs_shape, 2);

  std::vector<int64_t> inputs_len = {num_phones};
  const int64_t inputs_len_shape[] = {1};
  auto inputs_len_ort = Ort::Value::CreateTensor<int64_t>(
      encoder_.memory_info(), inputs_len.data(), inputs_len.size(),
      inputs_len_shape, 1);

  std::vector<float> scales = {0.667, 1.0, 0.8};
  const int64_t scales_shape[] = {1, 3};
  auto scales_ort = Ort::Value::CreateTensor<float>(
      encoder_.memory_info(), scales.data(), scales.size(), scales_shape, 2);

  std::vector<int64_t> spk_id = {sid};
  const int64_t spk_id_shape[] = {1};
  auto sid_ort = Ort::Value::CreateTensor<int64_t>(
      encoder_.memory_info(), spk_id.data(), spk_id.size(), spk_id_shape, 1);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  ort_inputs.push_back(std::move(inputs_len_ort));
  ort_inputs.push_back(std::move(scales_ort));
  ort_inputs.push_back(std::move(sid_ort));

  auto outputs_ort = encoder_.Run(ort_inputs);
  return outputs_ort;
}

void TtsModel::ForwardDecoder(Ort::Value&& z, const int sid,
                              std::vector<float>* audio) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<int64_t> spk_id = {sid};
  const int64_t spk_id_shape[] = {1};
  auto sid_ort = Ort::Value::CreateTensor<int64_t>(
      decoder_.memory_info(), spk_id.data(), spk_id.size(), spk_id_shape, 1);
  ort_inputs.push_back(std::move(z));
  ort_inputs.push_back(std::move(sid_ort));
  auto outputs_ort = decoder_.Run(ort_inputs);
  int len = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[2];
  const float* output_data = outputs_ort[0].GetTensorData<float>();
  audio->assign(output_data, output_data + len);
  for (size_t i = 0; i < audio->size(); i++) {
    (*audio)[i] *= 32767.0;
  }
}

void TtsModel::Text2PhoneIds(const std::string& text,
                             std::vector<int64_t>* phone_ids) {
  phone_ids->clear();
  // 1. TN
  std::string norm_text = tn_->Normalize(text);
  // 2. G2P: char => pinyin => phones => ids
  std::vector<std::string> phonemes;
  g2p_prosody_->Compute(norm_text, &phonemes);
  // 3. Convert to phone id
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
}

void TtsModel::Synthesis(const std::string& text, const int sid,
                         std::vector<float>* audio) {
  std::vector<int64_t> inputs;
  Text2PhoneIds(text, &inputs);
  auto outputs = ForwardEncoder(inputs, sid);
  ForwardDecoder(std::move(outputs[0]), sid, audio);
}

// See wetts/vits/inference_onnx.py
void TtsModel::SplitToChunks(const Ort::Value& z) {
  CHECK_GT(s_chunk_size_, 0);
  s_z_chunks_.clear();
  auto shape = z.GetTensorTypeAndShapeInfo().GetShape();
  const float* z_data = z.GetTensorData<float>();
  int mel_len = shape[1];
  hidden_dim_ = shape[2];
  int num = std::ceil(static_cast<float>(mel_len) / s_chunk_size_);
  for (int i = 0; i < num; i++) {
    int start = std::max(0, i * s_chunk_size_ - s_pad_size_);
    int end = std::min((i + 1) * s_chunk_size_ + s_pad_size_, mel_len);
    std::vector<float> chunk(z_data + start * hidden_dim_,
                             z_data + end * hidden_dim_);
    s_z_chunks_.push_back(std::move(chunk));
  }
}

// See wetts/vits/inference_onnx.py
void TtsModel::Depadding(int chunk_id, int num_chunks, int chunk_size,
                         int pad_size, int upsample,
                         std::vector<float>* audio) {
  int front_pad = std::min(chunk_id * chunk_size, pad_size);
  if (chunk_id == 0) {
    audio->assign(audio->begin(), audio->begin() + chunk_size * upsample);
  } else if (chunk_id == num_chunks - 1) {
    audio->assign(audio->begin() + front_pad * upsample, audio->end());
  } else {
    audio->assign(audio->begin() + front_pad * upsample,
                  audio->begin() + (front_pad + chunk_size) * upsample);
  }
}

bool TtsModel::StreamSynthesis(std::vector<float>* audio) {
  if (s_text_.empty()) return false;
  if (s_cur_ == 0) {
    std::vector<int64_t> inputs;
    Text2PhoneIds(s_text_, &inputs);
    auto outputs = ForwardEncoder(inputs, s_sid_);
    SplitToChunks(outputs[0]);
    VLOG(2) << "Split to chunks " << s_z_chunks_.size();
  }
  int num_chunks = s_z_chunks_.size();
  VLOG(2) << "Streaming decoder, progress " << s_cur_ << "/" << num_chunks;
  if (s_cur_ < num_chunks) {
    auto& chunk = s_z_chunks_[s_cur_];
    int num_mel = chunk.size() / hidden_dim_;
    const int64_t chunk_shape[] = {1, num_mel, hidden_dim_};
    auto z_chunk = Ort::Value::CreateTensor<float>(
        decoder_.memory_info(), chunk.data(), chunk.size(), chunk_shape, 3);
    ForwardDecoder(std::move(z_chunk), s_sid_, audio);
    if (s_chunk_size_ > 0) {
      Depadding(s_cur_, num_chunks, s_chunk_size_, s_pad_size_, kUpsampleRate,
                audio);
    }
  }
  s_cur_++;  // At least one chunk inference when calling
  return s_cur_ >= num_chunks ? false : true;
}

int TtsModel::GetSid(const std::string& name) {
  std::string default_sname = speaker2id_.begin()->first;
  if (speaker2id_.find(name) == speaker2id_.end()) {
    LOG(INFO) << "Invalid speaker name: " << name << ", ";
    LOG(INFO) << "fallback to default speaker: " << default_sname;
    return speaker2id_[default_sname];
  }
  return speaker2id_[name];
}

}  // namespace wetts
