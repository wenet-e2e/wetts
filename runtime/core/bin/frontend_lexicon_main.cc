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

#include "frontend/lexicon.h"

DEFINE_string(lexicon_file, "", "lexicon file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetts::Lexicon lexicon(FLAGS_lexicon_file);
  std::cout << lexicon.NumProns("和");
  std::cout << lexicon.Prons("和");

  std::cout << lexicon.NumProns("我");
  std::cout << lexicon.Prons("我");
  return 0;
}
