# Copyright (c) 2022, Yongqiang Li (yongqiangli@alumni.hust.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class T2S:
    def __init__(self, dict_file: str):
        self.t2p_dict = {}
        with open(dict_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                self.t2p_dict[line[0]] = line[1]

    def convert(self, x: str):
        x = list(x)
        for i in range(len(x)):
            if x[i] in self.t2p_dict:
                x[i] = self.t2p_dict.get(x[i])
        return ''.join(x)
