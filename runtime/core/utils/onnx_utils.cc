// Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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

#include "utils/onnx_utils.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

namespace wetts {

std::shared_ptr<Ort::Session> OnnxCreateSession(
    const std::string& model,
    const Ort::SessionOptions& session_options,
    Ort::Env* env) {
  try {
#ifdef _MSC_VER
    auto session = std::make_shared<Ort::Session>(
        *env, ToWString(model).c_str(), session_options);
#else
    auto session = std::make_shared<Ort::Session>(
        *env, model.c_str(), session_options);
    return session;
#endif
  } catch (std::exception const& e) {
    LOG(FATAL) << "error when load onnx model: " << e.what();
  }
  return nullptr;
}


void OnnxGetInputsOutputs(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names) {
  Ort::AllocatorWithDefaultOptions allocator;
  // Input info
  int num_nodes = session->GetInputCount();
  in_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetInputName(i, allocator);
    Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tInput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*in_names)[i] = name;
  }
  // Output info
  num_nodes = session->GetOutputCount();
  out_names->resize(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    char* name = session->GetOutputName(i, allocator);
    Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::vector<int64_t> node_dims = tensor_info.GetShape();
    std::stringstream shape;
    for (auto j : node_dims) {
      shape << j;
      shape << " ";
    }
    LOG(INFO) << "\tOutput " << i << " : name=" << name << " type=" << type
              << " dims=" << shape.str();
    (*out_names)[i] = name;
  }
}

}
