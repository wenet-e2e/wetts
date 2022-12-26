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

#include "http/http_server.h"

#include <vector>

#include "boost/beast/core.hpp"
#include "boost/beast/version.hpp"
#include "boost/url/src.hpp"
#include "boost/uuid/uuid.hpp"
#include "boost/uuid/uuid_generators.hpp"
#include "boost/uuid/uuid_io.hpp"

#include "frontend/wav.h"
#include "utils/string.h"

namespace wetts {

namespace urls = boost::urls;
namespace uuids = boost::uuids;

http::message_generator ConnectionHandler::handle_request(
    const std::string& wav_path) {
  // Attempt to open the file
  beast::error_code ec;
  http::file_body::value_type body;
  body.open(wav_path.c_str(), beast::file_mode::scan, ec);

  // Cache the size since we need it after the move
  auto const size = body.size();
  // Respond to GET request
  http::response<http::file_body> res{
      std::piecewise_construct, std::make_tuple(std::move(body)),
      std::make_tuple(http::status::ok, request_.version())};
  res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
  res.set(http::field::content_type, "audio/wav");
  res.content_length(size);
  res.keep_alive(request_.keep_alive());
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
    std::string sid = "0";
    if(params.contains("name"))
    {
      std::string name = (*params.find("name")).value;
      sid = tts_model_->Getsid(name);
      if(sid == ""){
        LOG(INFO) << "Unsupported speaker: " << name <<", use default speaker!";
        sid = "0";
      }
    }
    // 2. Synthesis audio from text
    std::vector<float> audio;
    tts_model_->Synthesis(text, sid, &audio);
    wetts::WavWriter wav_writer(audio.data(), audio.size(), 1, 22050, 16);
    // 3. Write samples to file named uuid.wav
    std::string wav_path =
        uuids::to_string(uuids::random_generator()()) + ".wav";
    wav_writer.Write(wav_path);

    // Handle request
    http::message_generator msg = handle_request(wav_path);
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
