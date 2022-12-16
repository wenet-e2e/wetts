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

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "http/http_server.h"
#include "utils/log.h"

DEFINE_string(tagger_file, "", "tagger fst file");
DEFINE_string(verbalizer_file, "", "verbalizer fst file");
DEFINE_string(g2p_prosody_model, "", "g2p prosody model file");
DEFINE_string(phone_file, "", "phone list file");
DEFINE_string(tokenizer_vocab_file, "", "tokenizer vocab file");
DEFINE_string(lexicon_file, "", "lexicon file");
DEFINE_string(e2e_model_file, "", "e2e tts model file");

DEFINE_int32(port, 10086, "http listening port");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto tn = std::make_shared<wetext::Processor>(FLAGS_tagger_file,
                                                FLAGS_verbalizer_file);
  auto g2p_prosody = std::make_shared<wetts::G2pProsody>(
      FLAGS_g2p_prosody_model, FLAGS_phone_file, FLAGS_tokenizer_vocab_file,
      FLAGS_lexicon_file);
  auto tts_model =
      std::make_shared<wetts::TtsModel>(FLAGS_e2e_model_file, tn, g2p_prosody);

  wetts::HttpServer server(FLAGS_port, tts_model);
  LOG(INFO) << "Listening at port " << FLAGS_port;
  server.Start();
  return 0;
}
