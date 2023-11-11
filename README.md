# WeTTS

Production First and Production Ready End-to-End Text-to-Speech Toolkit

## Install

### Install python package
``` sh
pip install git+https://github.com/wenet-e2e/wetts.git
```
**Command-line usage** (use `-h` for parameters):

``` sh
wetts --text "今天天气怎么样" --wav output.wav
```

**Python programming usage**:

``` python
import wetts

# TODO
```

### Install for development & deployment

We suggest to install WeTTS with Anaconda or Miniconda.

Clone this repo:

```sh
git clone https://github.com/wenet-e2e/wetts.git
```

Create the environment:

```bash
conda create -n wetts python=3.8 -y
conda activate wetts
pip install -r requirements.txt
```

## Roadmap

We mainly focus on end to end, production, and on-device TTS. We are going to use:

* backend: end to end model, such as:
  * [VITS](https://arxiv.org/pdf/2106.06103.pdf)
* frontend:
  * Text Normalization: [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
  * Prosody & Polyphones: [Unified Mandarin TTS Front-end Based on Distilled BERT Model](https://arxiv.org/pdf/2012.15404.pdf)

## Dataset

We plan to support a variaty of open source TTS datasets, include but not limited to:

* [Baker](https://www.data-baker.com/data/index/TNtts), Chinese Standard Mandarin Speech corpus open sourced by Data Baker.
* [AISHELL-3](https://openslr.org/93), a large-scale and high-fidelity multi-speaker Mandarin speech corpus.
* [Opencpop](https://wenet.org.cn/opencpop), Mandarin singing voice synthesis (SVS) corpus open sourced by Netease Fuxi.

## Pretrained Models

| Dataset        | Language | Checkpoint Model | Runtime Model |
| -------------- | -------- | ---------------- | ------------- |
| Baker          | CN       | [BERT](https://wenet.org.cn/downloads?models=wetts&version=baker_bert_exp.tar.gz) | [BERT](https://wenet.org.cn/downloads?models=wetts&version=baker_bert_onnx.tar.gz) |
| Multilingual   | CN       | [VITS](https://wenet.org.cn/downloads?models=wetts&version=multilingual_vits_v3_exp.tar.gz) | [VITS](https://wenet.org.cn/downloads?models=wetts&version=multilingual_vits_v3_onnx.tar.gz) |

## Runtime

We plan to support a variaty of hardwares and platforms, including:

* x86
* Android
* Raspberry Pi
* Other on-device platforms

``` bash
export GLOG_logtostderr=1
export GLOG_v=2

cd runtime/onnxruntime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/bin/tts_main \
  --frontend_flags baker_bert_onnx/frontend.flags \
  --vits_flags multilingual_vits_v3_onnx/vits.flags \
  --sname baker \
  --text "hello我是小明。" \
  --wav_path audio.wav
```

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
