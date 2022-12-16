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

#ifndef HTTP_HTTP_SERVER_H_
#define HTTP_HTTP_SERVER_H_

#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "boost/asio/ip/tcp.hpp"
#include "boost/beast/http.hpp"

#include "model/tts_model.h"

namespace wetts {

namespace beast = boost::beast;    // from <boost/beast.hpp>
namespace http = beast::http;      // from <boost/beast/http.hpp>
namespace asio = boost::asio;      // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;  // from <boost/asio/ip/tcp.hpp>

class ConnectionHandler {
 public:
  ConnectionHandler(tcp::socket&& socket, std::shared_ptr<TtsModel> tts_model)
      : socket_(std::move(socket)), tts_model_(std::move(tts_model)) {}
  void operator()();
  http::message_generator handle_request(const std::string& wav_path);

 private:
  tcp::socket socket_;
  http::request<http::string_body> request_;
  std::shared_ptr<TtsModel> tts_model_;
};

class HttpServer {
 public:
  HttpServer(int port, std::shared_ptr<TtsModel> tts_model)
      : port_(port), tts_model_(tts_model) {}

  void Start();

 private:
  int port_;
  // The io_context is required for all I/O
  asio::io_context ioc_{1};
  std::shared_ptr<TtsModel> tts_model_;
};

}  // namespace wetts

#endif  // HTTP_HTTP_SERVER_H_
