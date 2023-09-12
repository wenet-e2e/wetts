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

import os
import sys

if len(sys.argv) != 4:
    print("Usage: prepare_data.py lexicon in_data_dir out_data")
    sys.exit(-1)

lexicon = {}
with open(sys.argv[1], "r", encoding="utf8") as fin:
    for line in fin:
        arr = line.strip().split()
        lexicon[arr[0]] = arr[1:]

train_set_label_file = os.path.join(sys.argv[2], "train", "label_train-set.txt")
with open(train_set_label_file, encoding="utf8") as fin, open(
    sys.argv[3], "w", encoding="utf8"
) as fout:
    # skip the first five lines in label_train-set.txt
    lines = [x.strip() for x in fin.readlines()][5:]
    for line in lines:
        key, text, _ = line.split("|")
        speaker = key[:-4]
        wav_path = os.path.join(
            sys.argv[2], "train", "wav", speaker, "{}.wav".format(key)
        )
        phones = []
        for x in text.split():
            if x == "%" or x == "$":
                phones.append(x)
            elif x in lexicon:
                phones.extend(lexicon[x])
            else:
                print("{} OOV {}".format(key, x))
                sys.exit(-1)
        fout.write("{}|{}|sil {} sil\n".format(wav_path, speaker, " ".join(phones)))
