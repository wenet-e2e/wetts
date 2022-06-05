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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wav', type=str, help='Path to wav.txt.')
    parser.add_argument('speaker', type=str, help='Path to speaker.txt.')
    parser.add_argument('text', type=str, help='Path to text.txt.')
    parser.add_argument('duration', type=str, help='Path to duration.txt.')
    parser.add_argument('output_dir',
                        type=str,
                        help='Path to output directory.')
    parser.add_argument('--val_samples',
                        type=int,
                        default=20,
                        help='Number of validation samples for each speaker.')
    parser.add_argument('--test_samples',
                        type=str,
                        default=20,
                        help='Number of test samples for each speaker.')
    return parser.parse_args()


def write_file(wav_path, speaker_path, text_path, duration_path, samples):
    with open(wav_path, 'w') as fout:
        fout.writelines([x[0] for x in samples])
    with open(speaker_path, 'w') as fspeaker:
        fspeaker.writelines([x[1] for x in samples])
    with open(text_path, 'w') as ftext:
        ftext.writelines([x[2] for x in samples])
    with open(duration_path, 'w') as fduration:
        fduration.writelines([x[3] for x in samples])


def main(args):
    """Spliting dataset to training set, validation set and test set for each
    speaker.
    """
    samples = collections.defaultdict(list)
    with open(args.wav) as fwav, open(args.speaker) as fspeaker, open(
            args.text) as ftext, open(args.duration) as fduration:
        for wav, speaker, text, duration in zip(fwav, fspeaker, ftext,
                                                fduration):
            samples[speaker.strip()].append((wav, speaker, text, duration))
    training_samples = []
    val_samples = []
    test_samples = []
    for speaker in samples:
        training_samples.extend(samples[speaker][args.val_samples +
                                                 args.test_samples:])
        val_samples.extend(
            samples[speaker][args.test_samples:args.val_samples +
                             args.test_samples])
        test_samples.extend(samples[speaker][:args.test_samples])

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_samples_dir = output_dir / 'train'
    val_samples_dir = output_dir / 'val'
    test_samples_dir = output_dir / 'test'
    train_samples_dir.mkdir(parents=True, exist_ok=True)
    val_samples_dir.mkdir(parents=True, exist_ok=True)
    test_samples_dir.mkdir(parents=True, exist_ok=True)
    write_file(train_samples_dir / 'train_wav.txt',
               train_samples_dir / 'train_speaker.txt',
               train_samples_dir / 'train_text.txt',
               train_samples_dir / 'train_duration.txt', training_samples)
    write_file(val_samples_dir / 'val_wav.txt',
               val_samples_dir / 'val_speaker.txt',
               val_samples_dir / 'val_text.txt',
               val_samples_dir / 'val_duration.txt', val_samples)
    write_file(test_samples_dir / 'test_wav.txt',
               test_samples_dir / 'test_speaker.txt',
               test_samples_dir / 'test_text.txt',
               test_samples_dir / 'test_duration.txt', test_samples)


if __name__ == "__main__":
    main(get_args())
