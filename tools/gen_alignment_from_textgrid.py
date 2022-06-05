#!/usr/bin/env python3
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#               2022 Binbin Zhang(binbzha@qq.com)
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
import pathlib
from typing import List, Set

import numpy as np
from praatio import textgrid

SILENCE_TOKEN = set(['sp', 'sil'])


# in MFA1.x, there are blank labels("") in the end, and maybe "sp" before it
# in MFA2.x, there are  blank labels("") in the begin and the end,
# while no "sp" and "sil" anymore, we replace it with "sil"
def readtg(tg_path):
    alignment = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    phones = []
    ends = []
    for interval in alignment.tierDict["phones"].entryList:
        phone = interval.label
        phones.append(phone)
        ends.append(interval.end)
    durations = np.diff(np.array(ends), prepend=0)
    assert len(durations) == len(phones)
    # merge  "" and sp in the end
    if phones[-1] == "" and len(phones) > 1 and phones[-2] == "sp":
        phones = phones[:-1]
        durations[-2] += durations[-1]
        durations = durations[:-1]
    # replace the last "sp" with "sil" in MFA1.x
    phones[-1] = "sil" if phones[-1] == "sp" else phones[-1]
    # replace the edge "" with "sil", replace the inner "" with "sp"
    new_phones = []
    for i, phn in enumerate(phones):
        if phn == "":
            if i in {0, len(phones) - 1}:
                new_phones.append("sil")
            else:
                new_phones.append("sp")
        else:
            new_phones.append(phn)
    phones = new_phones
    return phones, durations.tolist()


def insert_special_tokens(seq1: List[str], seq2: List[str],
                          special_tokens: Set) -> List[str]:
    """Inserting special tokens into MFA aligned phoneme sequence.

    MFA aligned phoneme sequences contains no special token but contains silence
    phonemes such as 'sp' and 'sil'. However, FastSpeech2 expects phoneme
    sequences containing special tokens. This function will insert special
    tokens into MFA aligned phoneme sequence.

    Args:
        seq1 (List[str]): Phoneme sequence containing special tokens.
        seq2 (List[str]): MFA aligned phoneme sequence.
        special_tokens (Set): Special token set.

    Raises:
        ValueError: Indicating an insertion failure.

    Returns:
        List[str]: MFA aligned phoneme sequence with special tokens inserted.
    """
    new_seq = []
    i, j = 0, 0
    while i < len(seq1) and j < len(seq2):
        if seq1[i] == seq2[j]:
            new_seq.append(seq1[i])
            i += 1
            j += 1
        else:
            if seq1[i] in special_tokens:
                # we meet a special token in seq1
                # just insert it into new_seq
                # and move i to skip it
                new_seq.append(seq1[i])
                i += 1
            elif seq2[j] in SILENCE_TOKEN:
                # we meet a sp or sil in seq2
                # insert it into new_seq and
                # skip it
                new_seq.append(seq2[j])
                j += 1
            else:
                # we have found out an inconsistent sample
                # phoneme sequence containing special tokens should be the same
                # as MFA aligned phoneme sequence when removing special tokens
                # and silence phonemes ('sp' and 'sil') from both two types of
                # phoneme sequences.
                raise ValueError(
                    '{} and {} are inconsistent at pos {} and {}'.format(
                        seq1, seq2, i, j))
    while i < len(seq1):
        new_seq.append(seq1[i])
        i += 1
    while j < len(seq2):
        new_seq.append(seq2[j])
        j += 1
    return new_seq


def get_args():
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")
    parser.add_argument("wav", type=str, help="Path to wav.txt.")
    parser.add_argument("speaker", type=str, help="Path to speaker.txt.")
    parser.add_argument("text", type=str, help="Path to text.txt.")
    parser.add_argument("special_tokens",
                        type=str,
                        help='Path to sepcial_token.txt.')
    parser.add_argument("text_grid",
                        type=str,
                        help='Path to directory containing TextGrid.')
    parser.add_argument('aligned_wav',
                        type=str,
                        help='Path to file saving path of aligned wav.')
    parser.add_argument('aligned_speaker',
                        type=str,
                        help='Path to file saving speaker of aligned wav.')
    parser.add_argument(
        "duration",
        type=str,
        help='Path to duration.txt for saving exported durations.')
    parser.add_argument(
        "aligned_text",
        type=str,
        help=('Path to aligned_text.txt for saving phoneme sequences, ',
              'which are merged from text.txt and TextGrid.'))
    return parser.parse_args()


def main(args):
    text_grid_dir = pathlib.Path(args.text_grid)
    with open(args.special_tokens) as fin:
        special_tokens = set([x.strip() for x in fin])
    with open(args.wav) as fwav, open(args.speaker) as fspeaker, open(
            args.text) as ftext:
        aligned_text = []
        durations = []
        aligned_wav = []
        aligned_speaker = []
        for wav_path, speaker, text in zip(fwav, fspeaker, ftext):
            wav_path, speaker, text = (pathlib.Path(wav_path.strip()),
                                       speaker.strip(), text.strip().split())
            text_grid_path = text_grid_dir / speaker / '{}.TextGrid'.format(
                wav_path.stem)
            # only wav having alignment will be saved
            if text_grid_path.exists():
                aligned_wav.append(str(wav_path))
                aligned_speaker.append(speaker)
                tg_phones, duration = readtg(text_grid_path)
                aligned_text.append(
                    insert_special_tokens(text, tg_phones, special_tokens))
                durations.append(['{:.2f}'.format(x) for x in duration])
            else:
                print('Missing alignment: {}'.format(str(text_grid_path)))
    with open(args.aligned_wav, 'w') as fout:
        fout.writelines([x + '\n' for x in aligned_wav])
    with open(args.aligned_speaker, 'w') as fout:
        fout.writelines([x + '\n' for x in aligned_speaker])
    with open(args.duration, 'w') as fout:
        fout.writelines([' '.join(x) + '\n' for x in durations])
    with open(args.aligned_text, 'w') as fout:
        fout.writelines([' '.join(x) + '\n' for x in aligned_text])


if __name__ == "__main__":
    main(get_args())
