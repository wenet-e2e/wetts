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

#ifndef MODEL_TTS_MODEL_H_
#define MODEL_TTS_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "processor/wetext_processor.h"

#include "frontend/g2p_prosody.h"

namespace wetts {

const int kUpsampleRate = 256;

class TtsModel {
 public:
  explicit TtsModel(const std::string& encoder_model_path,
                    const std::string& decodder_model_path,
                    const std::string& speaker2id, const std::string& phone2id,
                    const int sampling_rate,
                    std::shared_ptr<wetext::Processor> processor,
                    std::shared_ptr<G2pProsody> g2p_prosody,
                    int chunk_size = 40, int pad_size = 10);
  std::vector<Ort::Value> ForwardEncoder(const std::vector<int64_t>& phonemes,
                                         const int sid);
  void ForwardDecoder(Ort::Value&& z, const int sid, std::vector<float>* audio);
  // Non-stream synthesis, direct call
  void Synthesis(const std::string& text, const int sid,
                 std::vector<float>* audio);
  // Stream synthesis, first call SetStreamInfo to set sid & text
  void SetStreamInfo(const std::string& text, int sid) {
    s_sid_ = sid;
    s_text_ = text;
    s_cur_ = 0;
    s_z_chunks_.clear();
  }
  // Stream synthesis, then call StreamSynthesis
  // Return:
  //     true: synthesis ok, you should continue to call me
  //     false: synthesis finished, stop call
  bool StreamSynthesis(std::vector<float>* audio);
  void SplitToChunks(const Ort::Value& z);
  void Depadding(int chunk_id, int num_chunks, int chunk_size, int pad,
                 int upsample, std::vector<float>* audio);
  void Text2PhoneIds(const std::string& text, std::vector<int64_t>* phone_ids);
  int GetSid(const std::string& name);
  int sampling_rate() const { return sampling_rate_; }

 private:
  // VITS encoder & decoder
  OnnxModel encoder_, decoder_;
  // For stream synthesis
  std::string s_text_;     // stream input text
  int s_sid_;              // stream input sid
  int s_cur_ = 0;          // stream synthesis index
  int s_chunk_size_ = 40;  // stream decoder chunk size
  int s_pad_size_ = 10;    // stream decoder pad size
  int hidden_dim_ = 192;
  std::vector<std::vector<float>> s_z_chunks_;  // stream decoder z chunks
  int sampling_rate_;
  std::unordered_map<std::string, int> phone2id_;
  std::unordered_map<std::string, int> speaker2id_;
  std::shared_ptr<wetext::Processor> tn_;
  std::shared_ptr<G2pProsody> g2p_prosody_;
};

}  // namespace wetts

#endif  // MODEL_TTS_MODEL_H_
