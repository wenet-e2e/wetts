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

#ifndef MODEL_ONNX_MODEL_H_
#define MODEL_ONNX_MODEL_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace wetts {

class OnnxModel {
 public:
  static void InitEngineThreads(int num_threads = 1);
  explicit OnnxModel(const std::string& model_path);

 protected:
  static Ort::Env env_;
  static Ort::SessionOptions session_options_;

  std::shared_ptr<Ort::Session> session_ = nullptr;
  Ort::MemoryInfo memory_info_ =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

  std::vector<const char*> input_node_names_;
  std::vector<const char*> output_node_names_;
  std::vector<Ort::AllocatedStringPtr> input_allocated_strings_;
  std::vector<Ort::AllocatedStringPtr> output_allocated_strings_;
};

}  // namespace wetts

#endif  // MODEL_ONNX_MODEL_H_
