# Copyright (c) 2022 Tsinghua University(Jie Chen)
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
import collections
import pathlib
import os
from typing import Iterable


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir',
                        type=str,
                        help='Path to AISHELL-3 dataset')
    parser.add_argument('wav', type=str, help='Path to export paths of wavs.')
    parser.add_argument('speaker', type=str, help='Path to export speakers.')
    parser.add_argument('text', type=str, help='Path to export text of wavs.')
    return parser.parse_args()


def save_scp_files(wav_scp_path: os.PathLike, speaker_scp_path: os.PathLike,
                   text_scp_path: os.PathLike, content: Iterable[str]):
    wav_scp_path = pathlib.Path(wav_scp_path)
    speaker_scp_path = pathlib.Path(speaker_scp_path)
    text_scp_path = pathlib.Path(text_scp_path)

    wav_scp_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_scp_path.parent.mkdir(parents=True, exist_ok=True)
    text_scp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(wav_scp_path, 'w') as wav_scp_file:
        wav_scp_file.writelines([str(line[0]) + '\n' for line in content])
    with open(speaker_scp_path, 'w') as speaker_scp_file:
        speaker_scp_file.writelines([line[1] + '\n' for line in content])
    with open(text_scp_path, 'w') as text_scp_file:
        text_scp_file.writelines([line[2] + '\n' for line in content])


def main(args):
    dataset_dir = pathlib.Path(args.dataset_dir)

    with open(dataset_dir / 'train' /
              'label_train-set.txt') as train_set_label_file:
        # skip the first five lines in label_train-set.txt
        train_set_label = [
            x.strip() for x in train_set_label_file.readlines()
        ][5:]
    samples = collections.defaultdict(list)
    for line in train_set_label:
        sample_name, tokens, _ = line.split('|')
        speaker = sample_name[:-4]
        wav_path = dataset_dir / 'train' / 'wav' / speaker / '{}.wav'.format(
            sample_name)
        if wav_path.exists():
            samples[speaker].append((wav_path.absolute(), speaker, tokens))

    sample_list = []

    for speaker in sorted(samples):
        sample_list.extend(samples[speaker])

    save_scp_files(args.wav, args.speaker, args.text, sample_list)


if __name__ == "__main__":
    main(get_args())
