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

#include "model/onnx_tts_model.h"
#include "utils/string.h"

DEFINE_string(model_path, "", "model path");
DEFINE_string(phonemes, "", "input phonemes");
DEFINE_string(phone_dict, "", "phone dict path");
DEFINE_string(wav_path, "", "output wave path");

typedef struct WAV_HEADER {
  /* RIFF Chunk Descriptor */
  uint8_t RIFF[4] = {'R', 'I', 'F', 'F'};  // RIFF Header Magic header
  uint32_t ChunkSize;                      // RIFF Chunk Size
  uint8_t WAVE[4] = {'W', 'A', 'V', 'E'};  // WAVE Header
  /* "fmt" sub-chunk */
  uint8_t fmt[4] = {'f', 'm', 't', ' '};  // FMT header
  uint32_t Subchunk1Size = 16;            // Size of the fmt chunk
  uint16_t AudioFormat = 1;  // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM
                             // Mu-Law, 258=IBM A-Law, 259=ADPCM
  uint16_t NumOfChan = 1;    // Number of channels 1=Mono 2=Sterio
  uint32_t SamplesPerSec = 22050;    // Sampling Frequency in Hz
  uint32_t bytesPerSec = 22050 * 2;  // bytes per second
  uint16_t blockAlign = 2;           // 2=16-bit mono, 4=16-bit stereo
  uint16_t bitsPerSample = 16;       // Number of bits per sample
  /* "data" sub-chunk */
  uint8_t Subchunk2ID[4] = {'d', 'a', 't', 'a'};  // "data"  string
  uint32_t Subchunk2Size;                         // Sampled data length
} wav_hdr;

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
  model->Forward(inputs, &audio);

  wav_hdr wav;
  wav.ChunkSize = audio.size() * 2 + sizeof(wav_hdr) - 8;
  wav.Subchunk2Size = audio.size() * 2 + sizeof(wav_hdr) - 44;
  std::ofstream out(FLAGS_wav_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&wav), sizeof(wav));
  for (int i = 0; i < audio.size(); ++i) {
    int16_t d = (int16_t)(audio[i] * 32767);
    out.write(reinterpret_cast<char*>(&d), sizeof(int16_t));
  }
  return 0;
}
