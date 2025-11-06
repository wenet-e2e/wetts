// Copyright (c) 2025 Binbin Zhang (binbzha@qq.com)
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

#include "model/vits_model.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace wetts {

VitsModel::VitsModel(const std::string& encoder_model_path,
                     const std::string& decoder_model_path, int chunk_size,
                     int pad_size)
    : encoder_(encoder_model_path),
      decoder_(decoder_model_path),
      chunk_size_(chunk_size),
      pad_size_(pad_size) {
  // Fixme(Binbin): cpplint  Do not indent within a namespace.
}

std::vector<Ort::Value> VitsModel::ForwardEncoder(
    const std::vector<int64_t>& phonemes, int sid) {
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

  std::vector<float> scales = {0.667f, 1.0f, 0.8f};
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

void VitsModel::ForwardDecoder(Ort::Value&& z, int sid,
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
    (*audio)[i] *= 32767.0f;
  }
}

void VitsModel::Forward(const std::vector<int64_t>& phonemes, int sid,
                        std::vector<float>* audio) {
  auto outputs = ForwardEncoder(phonemes, sid);
  ForwardDecoder(std::move(outputs[0]), sid, audio);
}

// See wetts/vits/inference_onnx.py
void VitsModel::SplitToChunks(const Ort::Value& z) {
  CHECK_GT(chunk_size_, 0);
  z_chunks_.clear();
  auto shape = z.GetTensorTypeAndShapeInfo().GetShape();
  const float* z_data = z.GetTensorData<float>();
  int mel_len = shape[1];
  hidden_dim_ = shape[2];
  int num = std::ceil(static_cast<float>(mel_len) / chunk_size_);
  for (int i = 0; i < num; i++) {
    int start = std::max(0, i * chunk_size_ - pad_size_);
    int end = std::min((i + 1) * chunk_size_ + pad_size_, mel_len);
    std::vector<float> chunk(z_data + start * hidden_dim_,
                             z_data + end * hidden_dim_);
    z_chunks_.push_back(std::move(chunk));
  }
}

// See wetts/vits/inference_onnx.py
void VitsModel::Depadding(int chunk_id, int num_chunks, int chunk_size,
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

void VitsModel::SetInput(const std::vector<int64_t>& phonemes, int sid) {
  sid_ = sid;
  cur_ = 0;
  z_chunks_.clear();
  auto outputs = ForwardEncoder(phonemes, sid_);
  SplitToChunks(outputs[0]);
  VLOG(2) << "Split to chunks " << z_chunks_.size();
}

bool VitsModel::StreamDecode(std::vector<float>* audio) {
  int num_chunks = z_chunks_.size();
  VLOG(2) << "Streaming decoder, progress " << cur_ << "/" << num_chunks;
  if (cur_ < num_chunks) {
    auto& chunk = z_chunks_[cur_];
    int num_mel = chunk.size() / hidden_dim_;
    const int64_t chunk_shape[] = {1, num_mel, hidden_dim_};
    auto z_chunk = Ort::Value::CreateTensor<float>(
        decoder_.memory_info(), chunk.data(), chunk.size(), chunk_shape, 3);
    ForwardDecoder(std::move(z_chunk), sid_, audio);
    if (chunk_size_ > 0) {
      Depadding(cur_, num_chunks, chunk_size_, pad_size_, kUpsampleRate, audio);
    }
  }
  cur_++;  // At least one chunk inference when calling
  return cur_ >= num_chunks ? true : false;
}

}  // namespace wetts
