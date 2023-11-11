# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
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

import scipy.io.wavfile as wavfile

from wetts.cli.model import load_model


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--text', help='text to synthesis')
    parser.add_argument('--wav', help='output wav file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = load_model()
    phones, audio = model.synthesis(args.text)
    wavfile.write(args.wav, 16000, audio)
    print('{} => {}'.format(args.text, ' '.join(phones)))
    print('Succeed, see {}'.format(args.wav))


if __name__ == '__main__':
    main()
