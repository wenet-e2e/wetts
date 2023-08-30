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


class Hanzi2Pinyin:
    def __init__(self, dict_file: str):
        self.pinyin_dict = {}
        with open(dict_file) as f:
            for line in f.readlines():
                line = line.strip()
                self.pinyin_dict[line[0]] = line[2:].split(",")

    def get(self, x):
        assert x in self.pinyin_dict
        return self.pinyin_dict[x]

    def convert(self, x: str):
        pinyin = []
        for char in x:
            pinyin.append(self.pinyin_dict.get(char, "UNK"))
        return pinyin


def main():
    hanzi2pinyin = Hanzi2Pinyin("local/pinyin_dict.txt")
    string = "汉字转拼音实验"
    pinyin = hanzi2pinyin.convert(string)
    print(string)
    print(pinyin)


if __name__ == "__main__":
    main()
