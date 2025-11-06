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

#ifndef MODEL_VITS_MODEL_H_
#define MODEL_VITS_MODEL_H_

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "model/onnx_model.h"

namespace wetts {

const int kUpsampleRate = 256;

// Thin wrapper around VITS encoder/decoder ONNX models
class VitsModel {
 public:
  explicit VitsModel(const std::string& encoder_model_path,
                     const std::string& decoder_model_path, int chunk_size = 40,
                     int pad_size = 10);

  std::vector<Ort::Value> ForwardEncoder(const std::vector<int64_t>& phonemes,
                                         int sid);
  void ForwardDecoder(Ort::Value&& z, int sid, std::vector<float>* audio);
  // Nonstream call, first call ForwardEncoder, then call ForwardDecoder
  void Forward(const std::vector<int64_t>& phonemes, int sid,
               std::vector<float>* audio);

  // Stream call, first call ForwardEncoder, then call ForwardDecoder
  // chunk by chunk
  void SetInput(const std::vector<int64_t>& phonemes, int sid);
  // Stream decode, chunk by chunk
  // Return:
  //     true:  done, all chunk finished, stop call
  //     false: work in progress, you should continue to call me
  bool StreamDecode(std::vector<float>* audio);
  void SplitToChunks(const Ort::Value& z);
  void Depadding(int chunk_id, int num_chunks, int chunk_size, int pad,
                 int upsample, std::vector<float>* audio);

 private:
  // VITS encoder & decoder
  OnnxModel encoder_;
  OnnxModel decoder_;

  int hidden_dim_ = 192;
  int chunk_size_ = 40;                       // stream decoder chunk size
  int pad_size_ = 10;                         // stream decoder pad size
  int sid_;                                   // stream input sid
  int cur_ = 0;                               // stream synthesis index
  std::vector<std::vector<float>> z_chunks_;  // stream decoder z chunks
};

}  // namespace wetts

#endif  // MODEL_VITS_MODEL_H_
