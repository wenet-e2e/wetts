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

#ifndef FRONTEND_SANDHI_H_
#define FRONTEND_SANDHI_H_

#include <string>
#include <vector>

namespace wetts {

// Please refer:
// https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/paddlespeech/t2s/frontend/tone_sandhi.py
// // NO_LINT
void Sandhi(const std::string& word, std::vector<std::string>* pinyin);

}  // namespace wetts

#endif  // FRONTEND_SANDHI_H_
