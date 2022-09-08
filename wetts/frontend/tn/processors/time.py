# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from processors.processor import Processor

from pynini import string_file
from pynini.lib.pynutil import delete, insert


class Time(Processor):

    def __init__(self):
        super().__init__(name='time')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        h = string_file('data/time/hour.tsv')
        m = string_file('data/time/minute.tsv')
        s = string_file('data/time/second.tsv')
        noon = string_file('data/time/noon.tsv')

        tagger = (
            insert('hour: "') + h + insert('" ') + delete(':') +
            insert('minute: "') + m + insert('"') +
            (delete(':') + insert(' second: "') + s + insert('"')).ques +
            delete(' ').ques + (insert(' noon: "') + noon + insert('"')).ques)
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        noon = delete('noon: "') + self.SIGMA + delete('" ')
        hour = delete('hour: "') + self.SIGMA + delete('" ')
        minute = delete('minute: "') + self.SIGMA + delete('"')
        second = delete(' second: "') + self.SIGMA + delete('"')
        verbalizer = noon.ques + hour + minute + second.ques
        self.verbalizer = self.delete_tokens(verbalizer)
