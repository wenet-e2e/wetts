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

#ifndef FRONTEND_TOKENIZER_H_
#define FRONTEND_TOKENIZER_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace wetts {

// The Tokenizer is like:
// https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/bert/tokenization_bert.py#L137  // NOLINT
class Tokenizer {
 public:
  explicit Tokenizer(const std::string& vocab_file);
  int NumTokens() const { return vocab_.size(); }
  void Tokenize(const std::string& str,
                std::vector<std::string>* tokens,
                std::vector<int64_t>* token_ids) const;

 private:
  std::unordered_map<std::string, int> vocab_;
  std::string cls_token_ = "[CLS]";
  std::string sep_token_ = "[SEP]";
};

}  // namespace wetts

#endif  // FRONTEND_TOKENIZER_H_
