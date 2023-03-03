// Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
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

#include <pybind11/pybind11.h>

#include "api/wetts_api.h"

namespace py = pybind11;


PYBIND11_MODULE(_wetts, m) {
  m.doc() = "wetts pybind11 plugin";  // optional module docstring
  m.def("wetts_init", &wetts_init, py::return_value_policy::reference,
        "wetts init");
  m.def("wetts_free", &wetts_free, "wetts free");
  m.def("wetts_synthesis", &wetts_synthesis, "wetts synthesis");
  m.def("wetts_get_result", &wetts_get_result, py::return_value_policy::copy,
        "wetts get result");
  m.def("wetts_set_log_level", &wetts_set_log_level, "set log level");
}
