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

#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "frontend/t2s.h"

DEFINE_string(t2s_file, "", "traditional to simplified dictionary");
DEFINE_string(input_file, "", "input testing file");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetts::T2S t2s(FLAGS_t2s_file);
  std::ifstream is(FLAGS_input_file);
  std::string line;
  while (getline(is, line)) {
    std::cout << t2s.Convert(line) << "\n";
  }
  return 0;
}
