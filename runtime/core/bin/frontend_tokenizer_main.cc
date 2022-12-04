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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "frontend/tokenizer.h"

DEFINE_string(vocab_file, "", "tokenizer vocab file");
DEFINE_string(input_file, "", "input testing file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetts::Tokenizer tokenizer(FLAGS_vocab_file);
  std::ifstream is(FLAGS_input_file);
  std::string line;
  while (getline(is, line)) {
    std::vector<std::string> tokens;
    std::vector<int> token_ids;
    tokenizer.Tokenize(line, &tokens);
    tokenizer.Tokenize(line, &token_ids);
    CHECK_EQ(tokens.size(), token_ids.size());
    for (const auto& x : tokens) {
      std::cout << x << " ";
    }
    std::cout << "\n";

    for (const auto& x : token_ids) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
