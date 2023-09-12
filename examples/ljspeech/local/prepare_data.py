#!/user/bin/env python3

# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
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

import csv
import os
import sys

from tools.cleaners import english_cleaners

if len(sys.argv) != 3:
    print("Usage: prepare_data.py in_data_dir out_data")
    sys.exit(-1)

metadata = os.path.join(sys.argv[1], "metadata.csv")
with open(metadata) as fin, open(sys.argv[2], "w", encoding="utf8") as fout:
    for row in csv.reader(fin, delimiter="|"):
        wav_path = os.path.join(sys.argv[1], f"wavs/{row[0]}.wav")
        phones = english_cleaners(row[-1], use_prosody=False)
        fout.write("{}|ljspeech|sil {} sil\n".format(wav_path, " ".join(phones)))
