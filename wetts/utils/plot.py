# Copyright (c) 2022 Horizon Robtics. (authors: Jie Chen)
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
# Modified from FastSpeech2(https://github.com/ming024/FastSpeech2)

from matplotlib import pyplot as plt


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(nrows=len(data),
                             ncols=1,
                             figsize=(12, 3 * len(data)))
    if titles is None:
        titles = ["" for _ in range(len(data))]

    for i, mel in enumerate(data):
        # mel: (t,d)
        im = axes[i].imshow(mel.T,
                            origin="lower",
                            interpolation='none',
                            aspect='auto')
        axes[i].set_ylim(0, mel.shape[1])
        axes[i].set_title(titles[i])
        axes[i].set_anchor("W")
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    return fig
