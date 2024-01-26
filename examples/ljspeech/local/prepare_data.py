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

import argparse
import csv
import os

from tools.cleaners import english_cleaners


def get_args():
    parser = argparse.ArgumentParser(description="prepare data")
    parser.add_argument("--data_dir", required=True, help="input data dir")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--use_prosody", default=True, help="whether use prosody")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    metadata = os.path.join(args.data_dir, "metadata.csv")
    with open(metadata) as fin, open(args.output, "w", encoding="utf8") as fout:
        for row in csv.reader(fin, delimiter="|"):
            wav_path = os.path.join(args.data_dir, f"wavs/{row[0]}.wav")
            phones = english_cleaners(row[-1], args.use_prosody)
            fout.write("{}|ljspeech|sil {}\n".format(wav_path, " ".join(phones)))


if __name__ == "__main__":
    main()
