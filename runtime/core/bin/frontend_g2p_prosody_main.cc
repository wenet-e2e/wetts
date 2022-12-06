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

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "frontend/g2p_prosody.h"

DEFINE_string(phone_file, "", "phone list file");
DEFINE_string(tokenizer_vocab_file, "", "tokenizer vocab file");
DEFINE_string(lexicon_file, "", "lexicon file");
DEFINE_string(g2p_prosody_model, "", "g2p prosody model file");
DEFINE_string(input_file, "", "input testing file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetts::G2pProsody g2p_prosody(FLAGS_g2p_prosody_model,
                                FLAGS_phone_file,
                                FLAGS_tokenizer_vocab_file,
                                FLAGS_lexicon_file);
  std::ifstream is(FLAGS_input_file);
  std::string line;
  while (getline(is, line)) {
    std::vector<std::string> phonemes;
    std::vector<int> prosody;
    g2p_prosody.Compute(line, &phonemes, &prosody);
    std::cout << line << "\n";
    for (const auto& x : phonemes) {
      std::cout << x << " ";
    }
    std::cout << "\n";
    for (const auto& x : prosody) {
      std::cout << x << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
