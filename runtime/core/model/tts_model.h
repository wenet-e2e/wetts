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

#ifndef MODEL_TTS_MODEL_H_
#define MODEL_TTS_MODEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "processor/processor.h"

#include "frontend/g2p_prosody.h"

namespace wetts {

class TtsModel : public OnnxModel {
 public:
  explicit TtsModel(const std::string& model_path,
                    const std::string& speaker_tabel_path,
                    std::shared_ptr<wetext::Processor> processor,
                    std::shared_ptr<G2pProsody> g2p_prosody);
  void Forward(const std::vector<int64_t>& phonemes, const int sid,
               std::vector<float>* audio);
  void Synthesis(const std::string& text, const int sid,
                 std::vector<float>* audio);
  int GetSid(const std::string& name);

 private:
  int sampling_rate_;
  std::map<std::string, int> speaker2id_;
  std::shared_ptr<wetext::Processor> tn_;
  std::shared_ptr<G2pProsody> g2p_prosody_;
};

}  // namespace wetts

#endif  // MODEL_TTS_MODEL_H_
