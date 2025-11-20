// Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
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

#ifndef FRONTEND_SENTENCE_BREAK_H_
#define FRONTEND_SENTENCE_BREAK_H_

#include <string>
#include <vector>

namespace wetts {

// 将输入文本按句子边界切分。
// max_clause_len: 最大子句长度（按 UTF-8 字符计数），默认 64；0 表示不限制。
void SentenceSegement(const std::string& text,
                      std::vector<std::string>* sentences,
                      size_t max_clause_len = 32);

}  // namespace wetts

#endif  // FRONTEND_SENTENCE_BREAK_H_
