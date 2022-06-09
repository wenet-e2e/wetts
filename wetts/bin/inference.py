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

import sys


def main():
    if sys.argv[1] == 'fastspeech2':
        from wetts.bin import fastspeech2_inference
        fastspeech2_inference.main(fastspeech2_inference.get_args(
            sys.argv[2:]))
    else:
        raise ValueError("Wrong model name!")


if __name__ == '__main__':
    main()
