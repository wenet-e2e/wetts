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

#ifndef MODEL_TTS_H_
#define MODEL_TTS_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "frontend/g2p_prosody.h"
#include "frontend/tn.h"
#include "model/vits_model.h"

namespace wetts {

class TTS {
 public:
  explicit TTS(const std::string& encoder_model_path,
               const std::string& decodder_model_path,
               const std::string& speaker2id, const std::string& phone2id,
               const int sampling_rate,
               std::shared_ptr<TN> tn,
               std::shared_ptr<G2pProsody> g2p_prosody, int chunk_size = 40,
               int pad_size = 10);
  // Non-stream synthesis, direct call
  void Synthesis(const std::string& text, const int sid,
                 std::vector<float>* audio);
  // Stream synthesis, first call SetStreamInfo to set sid & text
  void SetStreamInfo(const std::string& text, int sid);
  // Stream synthesis, then call StreamSynthesis
  // Return:
  //     true: synthesis done, stop call
  //     false: synthesis work in progress, you should continue to call me
  bool StreamSynthesis(std::vector<float>* audio);
  void Text2PhoneIds(const std::string& text, std::vector<int64_t>* phone_ids);
  int GetSid(const std::string& name);
  int sampling_rate() const { return sampling_rate_; }

 private:
  VitsModel vits_;

  std::vector<std::string> text_arrs_;  // for streaming synthesis
  int cur_text_idx_ = 0;                // for streaming synthesis
  int sid_;                             // for streaming synthesis, default to 0
  bool new_text_ = true;
  int sampling_rate_;
  std::unordered_map<std::string, int> phone2id_;
  std::unordered_map<std::string, int> speaker2id_;
  std::shared_ptr<TN> tn_;
  std::shared_ptr<G2pProsody> g2p_prosody_;
};

}  // namespace wetts

#endif  // MODEL_TTS_H_
