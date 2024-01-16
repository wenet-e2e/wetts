// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <jni.h>

#include <string>
#include <thread>

#include "frontend/g2p_en.h"
#include "frontend/g2p_prosody.h"
#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "model/tts_model.h"
#include "processor/wetext_processor.h"

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
DEFINE_string(vits_model, "", "e2e tts model file");
DEFINE_int32(sampling_rate, 22050, "sampling rate of pcm");

namespace wetts {

std::shared_ptr<wetts::TtsModel> model;
std::string model_dir;  // NOLINT

void init(JNIEnv* env, jobject, jstring jModelDir) {
  model_dir = std::string(env->GetStringUTFChars(jModelDir, nullptr));
  std::string frontendFlags = model_dir + "/frontend/frontend.flags";
  std::string vitsFlags = model_dir + "/vits/vits.flags";
  gflags::ReadFromFlagsFile(frontendFlags, "", false);
  gflags::ReadFromFlagsFile(vitsFlags, "", true);
  std::string tagger = model_dir + "/" + FLAGS_tagger;
  std::string verbalizer = model_dir + "/" + FLAGS_verbalizer;
  std::string cmudict = model_dir + "/" + FLAGS_cmudict;
  std::string g2p_en_model = model_dir + "/" + FLAGS_g2p_en_model;
  std::string g2p_en_sym = model_dir + "/" + FLAGS_g2p_en_sym;
  std::string g2p_prosody_model = model_dir + "/" + FLAGS_g2p_prosody_model;
  std::string vocab = model_dir + "/" + FLAGS_vocab;
  std::string char2pinyin = model_dir + "/" + FLAGS_char2pinyin;
  std::string pinyin2id = model_dir + "/" + FLAGS_pinyin2id;
  std::string pinyin2phones = model_dir + "/" + FLAGS_pinyin2phones;
  std::string vits_model = model_dir + "/" + FLAGS_vits_model;
  std::string speaker2id = model_dir + "/" + FLAGS_speaker2id;
  std::string phone2id = model_dir + "/" + FLAGS_phone2id;

  auto tn = std::make_shared<wetext::Processor>(tagger, verbalizer);
  std::shared_ptr<wetts::G2pEn> g2p_en =
      std::make_shared<wetts::G2pEn>(cmudict, g2p_en_model, g2p_en_sym);
  auto g2p_prosody = std::make_shared<wetts::G2pProsody>(
      g2p_prosody_model, vocab, char2pinyin, pinyin2id, pinyin2phones, g2p_en);
  model = std::make_shared<wetts::TtsModel>(vits_model, speaker2id, phone2id, FLAGS_sampling_rate,
                                            tn, g2p_prosody);
}

void run(JNIEnv* env, jobject, jstring jText, jstring jSpeaker) {
  std::string text = std::string(env->GetStringUTFChars(jText, nullptr));
  std::string speaker = std::string(env->GetStringUTFChars(jSpeaker, nullptr));
  std::vector<float> audio;
  int sid = model->GetSid(speaker);
  model->Synthesis(text, sid, &audio);
  wetts::WavWriter wav_writer(audio.data(), audio.size(), 1,
                              FLAGS_sampling_rate, 16);
  wav_writer.Write(model_dir + "/audio.wav");
}
}  // namespace wetts

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("cn/org/wenet/wetts/Synthesis");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;)V", reinterpret_cast<void*>(wetts::init)},
      {"run", "(Ljava/lang/String;Ljava/lang/String;)V",
       reinterpret_cast<void*>(wetts::run)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
