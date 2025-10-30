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

#include <memory>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "processor/wetext_processor.h"

#include "frontend/g2p_en.h"
#include "frontend/g2p_prosody.h"
#include "frontend/wav.h"
#include "model/tts_model.h"
#include "utils/string.h"

// Flags
DEFINE_string(frontend_flags, "", "frontend flags file");
DEFINE_string(vits_flags, "", "vits flags file");

// Text Normalization
DEFINE_string(tagger, "", "tagger fst file");
DEFINE_string(verbalizer, "", "verbalizer fst file");

// Tokenizer
DEFINE_string(vocab, "", "tokenizer vocab file");

// G2P for English
DEFINE_string(cmudict, "", "cmudict for english words");
DEFINE_string(g2p_en_model, "", "english g2p fst model for oov");
DEFINE_string(g2p_en_sym, "", "english g2p symbol table for oov");

// G2P for Chinese
DEFINE_string(char2pinyin, "", "chinese character to pinyin");
DEFINE_string(pinyin2id, "", "pinyin to id");
DEFINE_string(pinyin2phones, "", "pinyin to phones");
DEFINE_string(g2p_prosody_model, "", "g2p prosody model file");

// VITS
DEFINE_string(speaker2id, "", "speaker to id");
DEFINE_string(phone2id, "", "phone to id");
DEFINE_string(vits_encoder_model, "", "vits encoder model path");
DEFINE_string(vits_decoder_model, "", "vits decoder model path");
DEFINE_int32(sampling_rate, 22050, "sampling rate of pcm");

DEFINE_string(sname, "", "speaker name");
DEFINE_string(text, "", "input text");
DEFINE_string(wav_path, "", "output wave path");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  gflags::ReadFromFlagsFile(FLAGS_frontend_flags, "", false);
  gflags::ReadFromFlagsFile(FLAGS_vits_flags, "", false);

  auto tn = std::make_shared<wetext::Processor>(FLAGS_tagger, FLAGS_verbalizer);

  bool has_en = !FLAGS_g2p_en_model.empty() && !FLAGS_g2p_en_sym.empty() &&
                !FLAGS_g2p_en_sym.empty();
  std::shared_ptr<wetts::G2pEn> g2p_en =
      has_en ? std::make_shared<wetts::G2pEn>(FLAGS_cmudict, FLAGS_g2p_en_model,
                                              FLAGS_g2p_en_sym)
             : nullptr;

  auto g2p_prosody = std::make_shared<wetts::G2pProsody>(
      FLAGS_g2p_prosody_model, FLAGS_vocab, FLAGS_char2pinyin, FLAGS_pinyin2id,
      FLAGS_pinyin2phones, g2p_en);
  auto model = std::make_shared<wetts::TtsModel>(
      FLAGS_vits_encoder_model, FLAGS_vits_decoder_model, FLAGS_speaker2id,
      FLAGS_phone2id, FLAGS_sampling_rate, tn, g2p_prosody);

  std::vector<float> audio;
  int sid = model->GetSid(FLAGS_sname);
  model->Synthesis(FLAGS_text, sid, &audio);

  wetts::WavWriter wav_writer(audio.data(), audio.size(), 1,
                              FLAGS_sampling_rate, 16);
  wav_writer.Write(FLAGS_wav_path);
  return 0;
}
