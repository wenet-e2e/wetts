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
```
Install MFA:
```bash
conda install -n wetts montreal-forced-aligner=2.0.1 -c conda-forge -y
```
For CUDA 10.2, run:
``` sh
conda install -n wetts pytorch=1.11 torchaudio cudatoolkit=10.2 -c pytorch -y
```
For CUDA 11.3, run:
``` sh
conda install -n wetts pytorch=1.11 torchaudio cudatoolkit=11.3 -c pytorch -y
```
Installing other dependencies using:
```sh
conda activate wetts
python -m pip install -r requirements.txt
```

## Roadmap

We mainly focus on production and on-device TTS, and we plan to use:

* AM: FastSpeech2
* vocoder: hifigan/melgan

And we are going to provide reference solution of:

* Prosody
* Polyphones
* Text Normalization

## Dataset

We plan to support a variaty of open source TTS datasets, include but not limited to:

* [BZNSYP](https://www.data-baker.com/data/index/TNtts/), Chinese Standard Mandarin Speech corpus open sourced by Data Baker.
* [AISHELL-3](https://openslr.org/93/), a large-scale and high-fidelity multi-speaker Mandarin speech corpus.
* [Opencpop](https://wenet.org.cn/opencpop/), Mandarin singing voice synthesis (SVS) corpus open sourced by Netease Fuxi.

## Runtime

We plan to support a variaty of hardwares and platforms, including:

* x86
* Android
* Raspberry Pi
* Other on-device platforms

## Acknowledgement

1. We borrow some code from [FastSpeech2](https://github.com/ming024/FastSpeech2) for FastSpeech2 implentation.
2. We refer [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) for feature extraction,
   `pinyin` lexicon preparation for alignment.
3. we borrowed some code from [chinese test normalization](https://github.com/speechio/chinese_text_normalization) for text normalization.
