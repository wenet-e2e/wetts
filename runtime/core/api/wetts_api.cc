// Copyright (c) 2022  Binbin Zhang (binbzha@qq.com)
//               2023  Xingchen Song(sxc19@mails.tsinghua.edu.cn)
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

#include "api/wetts_api.h"

#include <memory>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "processor/processor.h"
#include "frontend/g2p_prosody.h"
#include "frontend/wav.h"
#include "utils/string.h"
#include "utils/file.h"
#include "model/tts_model.h"

class Synthesiser {
 public:
  explicit Synthesiser(const std::string& model_dir) {
    // tn init
    std::string tagger_file = wetts::JoinPath(model_dir, "tagger.fst");
    std::string verbalizer_file = wetts::JoinPath(model_dir, "verbalizer.fst");
    CHECK(wetts::FileExists(tagger_file));
    CHECK(wetts::FileExists(verbalizer_file));
    tn_ = std::make_shared<wetext::Processor>(tagger_file, verbalizer_file);

    // g2p_prosody init
    std::string g2p_model_file = wetts::JoinPath(model_dir, "tagger.fst");
    std::string phone_file = wetts::JoinPath(model_dir, "tagger.fst");
    std::string vocab_file = wetts::JoinPath(model_dir, "tagger.fst");
    std::string lexicon_file = wetts::JoinPath(model_dir, "tagger.fst");
    CHECK(wetts::FileExists(g2p_model_file));
    CHECK(wetts::FileExists(phone_file));
    CHECK(wetts::FileExists(vocab_file));
    CHECK(wetts::FileExists(lexicon_file));
    g2p_prosody_ = std::make_shared<wetts::G2pProsody>(
      g2p_model_file, phone_file, vocab_file, lexicon_file);

    // tts_model init
    std::string tts_model_file = wetts::JoinPath(model_dir, "tagger.fst");
    std::string speaker_table = wetts::JoinPath(model_dir, "tagger.fst");
    CHECK(wetts::FileExists(tts_model_file));
    CHECK(wetts::FileExists(speaker_table));
    tts_model_ = std::make_shared<wetts::TtsModel>(
        tts_model_file, speaker_table, tn_, g2p_prosody_);
  }

  void Synthesis(const char* text, const char* sname) {
    std::string tmp_text(text), tmp_sname(sname);
    int sid = tts_model_->GetSid(tmp_sname);
    tts_model_->Synthesis(tmp_text, sid, &audio_);
  }

  const float* GetResult() { return audio_.data(); }
  void SetLanguage(const char* lang) { language_ = lang; }

 private:
  std::shared_ptr<wetext::Processor> tn_ = nullptr;
  std::shared_ptr<wetts::G2pProsody> g2p_prosody_ = nullptr;
  std::shared_ptr<wetts::TtsModel> tts_model_ = nullptr;
  std::shared_ptr<wetts::WavWriter> wav_writer_ = nullptr;

  std::string language_ = "chs";  // currently not used.
  std::vector<float> audio_;
};

void* wetts_init(const char* model_dir) {
  Synthesiser* synthesiser = new Synthesiser(model_dir);
  return reinterpret_cast<void*>(synthesiser);
}

void wetts_free(void* synthesiser) {
  delete reinterpret_cast<Synthesiser*>(synthesiser);
}

void wetts_synthesis(void* synthesiser, const char* text, const char* sname) {
  Synthesiser* synthesiser = reinterpret_cast<Synthesiser*>(synthesiser);
  synthesiser->Synthesis(text, sname);
}

const float* wetts_get_result(void* synthesiser) {
  Synthesiser* synthesiser = reinterpret_cast<Synthesiser*>(synthesiser);
  return synthesiser->GetResult();
}

// TODO(xcsong): support multi-language, currently it is hard code to "chs"
void wetts_set_language(void* synthesiser, const char* lang) {
  Synthesiser* synthesiser = reinterpret_cast<Synthesiser*>(synthesiser);
  synthesiser->SetLanguage(lang);
}

void wetts_set_log_level(int level) {
  FLAGS_logtostderr = true;
  FLAGS_v = level;
}
