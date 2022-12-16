// Copyright (c) 2022, Binbin Zhang (binbzha@qq.com)
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

#ifndef FRONTEND_G2P_PROSODY_H_
#define FRONTEND_G2P_PROSODY_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

#include "frontend/lexicon.h"
#include "frontend/tokenizer.h"
#include "model/onnx_model.h"

namespace wetts {

// Unified G2P & Prosody model
class G2pProsody : public OnnxModel {
 public:
  explicit G2pProsody(const std::string& g2p_prosody_model,
                      const std::string& phone_file,
                      const std::string& tokenizer_vocab_file,
                      const std::string& lexicon_file);
  void Compute(const std::string& str, std::vector<std::string>* phonemes,
               std::vector<int>* prosody);

 private:
  std::vector<std::string> phones_;
  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<Lexicon> lexicon_;
};

}  // namespace wetts

#endif  // FRONTEND_G2P_PROSODY_H_
