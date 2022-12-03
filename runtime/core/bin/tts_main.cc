// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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
#include <unordered_map>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "frontend/wav.h"
#include "model/onnx_tts_model.h"
#include "utils/string.h"

DEFINE_string(model_path, "", "model path");
DEFINE_string(phonemes, "", "input phonemes");
DEFINE_string(phone_dict, "", "phone dict path");
DEFINE_string(wav_path, "", "output wave path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetts::OnnxTtsModel::InitEngineThreads(1);
  auto model = std::make_shared<wetts::OnnxTtsModel>();
  model->Read(FLAGS_model_path);

  std::unordered_map<std::string, int64_t> phones_map;
  std::ifstream file(FLAGS_phone_dict);
  std::string line;
  while (getline(file, line)) {
    std::vector<std::string> strs;
    wetts::SplitString(line, &strs);
    phones_map[strs[0]] = std::stoi(strs[1]);
  }

  std::vector<std::string> phones;
  wetts::SplitString(FLAGS_phonemes, &phones);
  std::vector<int64_t> inputs;
  for (std::string phone : phones) {
    CHECK_NE(phones_map.count(phone), 0);
    inputs.emplace_back(phones_map[phone]);
  }

  std::vector<float> audio;
  model->Forward(&inputs, &audio);
  for (size_t i = 0; i < audio.size(); i++) {
    audio[i] *= 32767.0;
  }

  wetts::WavWriter wav_writer(audio.data(), audio.size(), 1, 22050, 16);
  wav_writer.Write(FLAGS_wav_path);
  return 0;
}
