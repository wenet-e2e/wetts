// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//               2024 Shengqiang Li (shengqiang.li96@gmail.com)
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

#include "http/http_server.h"

#include <vector>

#include "boost/beast/core.hpp"
#include "boost/beast/core/detail/base64.hpp"
#include "boost/beast/version.hpp"
#include "boost/url/src.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"
#include "json/json.h"

#include "frontend/wav.h"
#include "utils/string.h"
#include "utils/timer.h"

namespace wetts {

namespace urls = boost::urls;
namespace uuids = boost::uuids;
namespace base64 = boost::beast::detail::base64;

http::message_generator ConnectionHandler::HandleRequest(
    const std::string& json_data) {
  beast::error_code ec;
  http::response<http::string_body> res;
  res.result(http::status::ok);
  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "application/json");
  res.keep_alive(request_.keep_alive());
  res.body() = json_data;
  res.prepare_payload();
  return res;
}

void ConnectionHandler::operator()() {
  beast::error_code ec;
  // This buffer is required to persist across reads
  beast::flat_buffer buffer;
  for (;;) {
    // Read a request
    http::read(socket_, buffer, request_, ec);
    if (ec == http::error::end_of_stream) break;
    if (ec) {
      LOG(ERROR) << "read: " << ec.message();
      return;
    }

    // 1. Parse params from requests target
    urls::result<urls::url_view> rv =
        urls::parse_uri_reference(request_.target());
    if (!rv) {
      LOG(ERROR) << "Invalid requests target: " << rv.error();
      break;
    }
    urls::params_view params = rv->params();
    if (!params.contains("text")) break;
    std::string text = (*params.find("text")).value;
    LOG(INFO) << "input text: " << text;
    // speaker id
    if (!params.contains("name")) break;
    std::string name = (*params.find("name")).value;
    int sid = tts_model_->GetSid(name);
    // 2. Synthesis audio from text
    int sample_rate = tts_model_->sampling_rate();
    int num_channels = 1;
    int bits_per_sample = 16;
    LOG(INFO) << "Sample rate: " << sample_rate;
    LOG(INFO) << "Num of channels: " << num_channels;
    LOG(INFO) << "Bit per sample: " << bits_per_sample;
    int extract_time = 0;
    wetts::Timer timer;
    std::vector<float> pcm;
    tts_model_->Synthesis(text, sid, &pcm);
    int pcm_size = pcm.size();
    extract_time = timer.Elapsed();
    LOG(INFO) << "TTS pcm duration: "
              << pcm_size * 1000 / num_channels / sample_rate << "ms";
    LOG(INFO) << "Cost time: " << static_cast<float>(extract_time) << "ms";
    // 3. Convert pcm to wav
    std::vector<int16_t> audio(pcm_size);
    for (int i = 0; i < pcm_size; ++i) {
      audio[i] = static_cast<int16_t>(pcm[i]);
    }
    int audio_size = pcm_size * sizeof(int16_t);
    int data_size = audio_size + 44;
    WavHeader header(pcm_size, num_channels, sample_rate, bits_per_sample);
    std::vector<char> wav_data;
    wav_data.insert(wav_data.end(), reinterpret_cast<char*>(&header),
                    reinterpret_cast<char*>(&header) + 44);
    wav_data.insert(wav_data.end(), reinterpret_cast<char*>(audio.data()),
                    reinterpret_cast<char*>(audio.data()) + audio_size);
    // 4. Base64 encode
    Json::Value response;
    std::size_t encode_size = base64::encoded_size(data_size);
    std::vector<char> base64_wav(encode_size);
    std::size_t encoded_size =
        base64::encode(base64_wav.data(), wav_data.data(), data_size);
    std::string encoded_wav(base64_wav.begin(), base64_wav.end());
    response["audio"] = encoded_wav;
    std::string json_data = response.toStyledString();
    // Handle request
    http::message_generator msg = HandleRequest(json_data);
    // Determine if we should close the connection
    bool keep_alive = msg.keep_alive();
    // Send the response
    beast::write(socket_, std::move(msg), ec);

    if (ec) {
      LOG(ERROR) << "write: " << ec.message();
      return;
    }
    if (!keep_alive) {
      // This means we should close the connection, usually because
      // the response indicated the "Connection: close" semantic.
      break;
    }
  }
  // Send a TCP shutdown
  socket_.shutdown(tcp::socket::shutdown_send, ec);
}

void HttpServer::Start() {
  auto const address = asio::ip::make_address("0.0.0.0");
  // The acceptor receives incoming connections
  tcp::acceptor acceptor{ioc_, {address, static_cast<uint16_t>(port_)}};
  for (;;) {
    // This will receive the new connection
    tcp::socket socket{ioc_};
    // Block until we get a connection
    acceptor.accept(socket);
    // Launch the session, transferring ownership of the socket
    ConnectionHandler handler(std::move(socket), tts_model_);
    std::thread t(std::move(handler));
    t.detach();
  }
}

}  // namespace wetts
