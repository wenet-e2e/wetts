// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

#include <utility>

#include "glog/logging.h"

#include "utils/string.h"

namespace wetts {

TtsModel::TtsModel(const std::string& model_path,
                   std::shared_ptr<wetext::Processor> tn,
                   std::shared_ptr<G2pProsody> g2p_prosody)
    : OnnxModel(model_path),
      tn_(std::move(tn)),
      g2p_prosody_(std::move(g2p_prosody)) {
  // TODO(zhendong.peng): Read metadata
  // auto model_metadata = session_->GetModelMetadata();
  // Ort::AllocatorWithDefaultOptions allocator;
  // sampling_rate_ =
  //     atoi(model_metadata.LookupCustomMetadataMap("sampling_rate",
  //     allocator));
  // LOG(INFO) << "Onnx Model Info:";
  // LOG(INFO) << "\tsampling_rate " << sampling_rate_;
}

void TtsModel::Forward(const std::vector<int64_t>& phonemes,
                       std::vector<float>* audio) {
  int num_phones = phonemes.size();
  const int64_t inputs_shape[] = {1, num_phones};
  auto inputs_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info_, const_cast<int64_t*>(phonemes.data()), num_phones,
      inputs_shape, 2);

  std::vector<int64_t> inputs_len = {num_phones};
  const int64_t inputs_len_shape[] = {1};
  auto inputs_len_ort = Ort::Value::CreateTensor<int64_t>(
      memory_info_, inputs_len.data(), inputs_len.size(), inputs_len_shape, 1);

  std::vector<float> scales = {0.667, 1.0, 0.8};
  const int64_t scales_shape[] = {1, 3};
  auto scales_ort = Ort::Value::CreateTensor<float>(
      memory_info_, scales.data(), scales.size(), scales_shape, 2);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(inputs_ort));
  ort_inputs.push_back(std::move(inputs_len_ort));
  ort_inputs.push_back(std::move(scales_ort));

  auto outputs_ort = session_->Run(
      Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
      ort_inputs.size(), output_node_names_.data(), 1);
  int len = outputs_ort[0].GetTensorTypeAndShapeInfo().GetShape()[2];
  const float* outputs = outputs_ort[0].GetTensorData<float>();
  audio->assign(outputs, outputs + len);
}

void TtsModel::Synthesis(const std::string& text, std::vector<float>* audio) {
  // 1. TN
  std::string norm_text = tn_->normalize(text);
  // 2. G2P: char => pinyin => phones => ids
  std::vector<std::string> pinyins;
  std::vector<int> prosody;
  g2p_prosody_->Compute(norm_text, &pinyins, &prosody);

  std::vector<int64_t> inputs;
  for (int i = 0; i < pinyins.size(); ++i) {
    std::vector<std::string> phonemes;
    SplitString(pinyins[i], &phonemes);
    for (std::string phoneme : phonemes) {
      inputs.emplace_back(std::stoi(phoneme));
    }
    if (prosody[i] != 0) {
      inputs.emplace_back(prosody[i]);
    }
  }

  Forward(inputs, audio);
  for (size_t i = 0; i < audio->size(); i++) {
    (*audio)[i] *= 32767.0;
  }
}

}  // namespace wetts
