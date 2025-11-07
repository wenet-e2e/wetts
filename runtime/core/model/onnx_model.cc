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

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model/onnx_model.h"

#include "utils/log.h"

#include "utils/string.h"

namespace wetts {

Ort::Env OnnxModel::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions OnnxModel::session_options_ = Ort::SessionOptions();

void OnnxModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
}

OnnxModel::OnnxModel(const std::string& model_path) {
  InitEngineThreads(1);
#ifdef _MSC_VER
  session_ = std::make_shared<Ort::Session>(env_, ToWString(model_path).c_str(),
                                            session_options_);
#else
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
#endif
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session_->GetInputCount();
  input_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto input_name = session_->GetInputNameAllocated(i, allocator);
    input_allocated_strings_.push_back(std::move(input_name));
    input_node_names_[i] = input_allocated_strings_[i].get();
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "Input " << i << " : name=" << input_node_names_[i]
              << " type=" << type << " dims=" << shape.str();
  }

  // Output info
  num_nodes = session_->GetOutputCount();
  output_node_names_.resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);
    output_allocated_strings_.push_back(std::move(output_name));
    output_node_names_[i] = output_allocated_strings_[i].get();
    Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "Output " << i << " : name=" << output_node_names_[i]
              << " type=" << type << " dims=" << shape.str();
  }
}

std::vector<Ort::Value> OnnxModel::Run(
    const std::vector<Ort::Value>& input_values) {
  return session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(),
                       input_values.data(), input_node_names_.size(),
                       output_node_names_.data(), output_node_names_.size());
}

}  // namespace wetts
