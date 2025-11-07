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

#ifndef FRONTEND_TN_H_
#define FRONTEND_TN_H_

#include <string>

#ifdef FST
#include "processor/wetext_processor.h"
#endif  //  FST

namespace wetts {

#ifdef FST
class TN {
 public:
  TN(const std::string& tagger_path, const std::string& verbalizer_path)
      : tn_(tagger_path, verbalizer_path) {}
  std::string Normalize(const std::string& text) {
    return tn_.Normalize(text);
  }

 private:
  wetext::Processor tn_;
};

#else
class TN {
 public:
  TN(const std::string& tagger_path, const std::string& verbalizer_path) {}
  std::string Normalize(const std::string& text) { return text; }
};

#endif  //  FST

}  // namespace wetts

#endif  // FRONTEND_TN_H_
