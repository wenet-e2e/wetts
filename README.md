# WeTTS

Production First and Production Ready End-to-End Text-to-Speech Toolkit

Note: This project is at its early statge now.
Its design and implementation are subjected to change.

## Install

We suggest installing WeTTS with Anaconda or Miniconda.
Clone this repo:
```sh
git clone https://github.com/wenet-e2e/wetts.git
```
Create environment:
```bash
conda create -n wetts python=3.8 -y
conda activate wetts
pip install -r requirements.txt
conda install -n wetts pytorch=1.11 torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge -y
```

Please note you should use `cudatoolkit=11.3` for CUDA 11.3.


## Roadmap

We mainly focus on end to end, production, and on-device TTS. We are going to use:

* backend: end to end model, such as:
  * [VITS](https://arxiv.org/pdf/2106.06103.pdf)
* frontend:
  * Text Normalization: [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
  * Prosody & Polyphones: [Unified Mandarin TTS Front-end Based on Distilled BERT Model](https://arxiv.org/pdf/2012.15404.pdf)

## Dataset

We plan to support a variaty of open source TTS datasets, include but not limited to:

* [baker](https://www.data-baker.com/data/index/TNtts/), Chinese Standard Mandarin Speech corpus open sourced by Data Baker.
* [AISHELL-3](https://openslr.org/93/), a large-scale and high-fidelity multi-speaker Mandarin speech corpus.
* [Opencpop](https://wenet.org.cn/opencpop/), Mandarin singing voice synthesis (SVS) corpus open sourced by Netease Fuxi.

## Runtime

We plan to support a variaty of hardwares and platforms, including:

* x86
* Android
* Raspberry Pi
* Other on-device platforms

## Discussion & Communication

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/pengzhendong/files/blob/master/images/wechat.png" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wetts/issues).

## Acknowledgement

1. We borrow a lot of code from [vits](https://github.com/jaywalnut310/vits) for VITS implementation.
2. We refer [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) for `pinyin` lexicon generation.
