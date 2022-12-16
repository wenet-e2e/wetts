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

#ifndef MODEL_ONNX_TTS_MODEL_H_
#define MODEL_ONNX_TTS_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "model/onnx_model.h"

namespace wetts {

class OnnxTtsModel : public OnnxModel {
 public:
  explicit OnnxTtsModel(const std::string& model_path);
  void Forward(std::vector<int64_t>* phonemes, std::vector<float>* audio);

 private:
  int sampling_rate_;
};

}  // namespace wetts

#endif  // MODEL_ONNX_TTS_MODEL_H_
