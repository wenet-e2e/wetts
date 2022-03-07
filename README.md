# WeTTS

Production First and Production Ready End-to-End Text-to-Speech Toolkit

## Install

``` sh
conda create -n aligner -c conda-forge montreal-forced-aligner
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

