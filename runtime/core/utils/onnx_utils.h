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

#ifndef UTILS_ONNX_UTILS_H_
#define UTILS_ONNX_UTILS_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace wetts {

std::shared_ptr<Ort::Session> OnnxCreateSession(
    const std::string& model,
    const Ort::SessionOptions& session_options,
    Ort::Env* env);


void OnnxGetInputsOutputs(const std::shared_ptr<Ort::Session>& session,
                          std::vector<const char*>* in_names,
                          std::vector<const char*>* out_names);

}

#endif  // UTILS_ONNX_UTILS_H_


