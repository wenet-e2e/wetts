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
#include "processor/processor.h"

#include "frontend/g2p_prosody.h"
#include "frontend/wav.h"
#include "model/onnx_tts_model.h"
#include "utils/string.h"

DEFINE_string(text, "", "input text");

DEFINE_string(tagger_file, "", "tagger fst file");
DEFINE_string(verbalizer_file, "", "verbalizer fst file");

DEFINE_string(g2p_prosody_model, "", "g2p prosody model file");
DEFINE_string(phone_file, "", "phone list file");
DEFINE_string(tokenizer_vocab_file, "", "tokenizer vocab file");
DEFINE_string(lexicon_file, "", "lexicon file");

DEFINE_string(e2e_model_file, "", "e2e tts model file");
DEFINE_string(wav_path, "", "output wave path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  wetext::Processor processor(FLAGS_tagger_file, FLAGS_verbalizer_file);
  wetts::G2pProsody g2p_prosody(FLAGS_g2p_prosody_model, FLAGS_phone_file,
                                FLAGS_tokenizer_vocab_file, FLAGS_lexicon_file);
  wetts::OnnxTtsModel::InitEngineThreads(1);
  auto model = std::make_shared<wetts::OnnxTtsModel>();
  model->Read(FLAGS_e2e_model_file);

  // 1. TN
  std::string normalized_text = processor.normalize(FLAGS_text);
  // 2. G2P: char => pinyin => phones => ids
  std::vector<std::string> pinyins;
  std::vector<int> prosody;
  g2p_prosody.Compute(normalized_text, &pinyins, &prosody);

  std::vector<int64_t> inputs;
  for (int i = 0; i < pinyins.size(); ++i) {
    std::vector<std::string> phonemes;
    wetts::SplitString(pinyins[i], &phonemes);
    for (std::string phoneme : phonemes) {
      inputs.emplace_back(std::stoi(phoneme));
    }
    if (prosody[i] != 0) {
      inputs.emplace_back(prosody[i]);
    }
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
