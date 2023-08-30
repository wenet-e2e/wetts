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

* [Baker](https://www.data-baker.com/data/index/TNtts/), Chinese Standard Mandarin Speech corpus open sourced by Data Baker.
* [AISHELL-3](https://openslr.org/93/), a large-scale and high-fidelity multi-speaker Mandarin speech corpus.
* [Opencpop](https://wenet.org.cn/opencpop/), Mandarin singing voice synthesis (SVS) corpus open sourced by Netease Fuxi.

## Pretrained Models

| Dataset | Language | Checkpoint Model | Runtime Model |
| ------- | -------- | ---------------- | ------------- |
| Baker   | CN       | [BERT](https://wenet.org.cn/downloads?models=wetts&version=baker_bert_exp.tar.gz) | [BERT](https://wenet.org.cn/downloads?models=wetts&version=baker_bert_onnx.tar.gz) |
| Baker   | CN       | [VITS](https://wenet.org.cn/downloads?models=wetts&version=baker_vits_v1_exp.tar.gz) | [VITS](https://wenet.org.cn/downloads?models=wetts&version=baker_vits_v1_onnx.tar.gz) |

English G2P model: [english_us_arpa v2.0.0a](https://wenet.org.cn/downloads?models=wetts&version=g2p_en.tar.gz), powered by [MFA](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/g2p-english_us_arpa-v2.0.0a).

## Runtime

We plan to support a variaty of hardwares and platforms, including:

* x86
* Android
* Raspberry Pi
* Other on-device platforms

``` bash
export GLOG_logtostderr=1
export GLOG_v=2

./build/bin/tts_main \
  --tagger baker_bert_onnx/zh_tn_tagger.fst \
  --verbalizer baker_bert_onnx/zh_tn_verbalizer.fst \
  --vocab baker_bert_onnx/vocab.txt \
  --char2pinyin baker_bert_onnx/pinyin_dict.txt \
  --pinyin2id baker_bert_onnx/polyphone_phone.txt \
  --pinyin2phones baker_bert_onnx/lexicon.txt \
  --g2p_prosody_model baker_bert_onnx/19.onnx \
  --speaker2id baker_vits_v1_onnx/speaker.txt \
  --sname baker \
  --phone2id baker_vits_v1_onnx/phones.txt \
  --vits_model baker_vits_v1_onnx/G_250000.onnx \
  --text "你好，我是小明。" \
  --cmudict g2p_en/cmudict.dict \  # optional
  --g2p_en_model g2p_en/model.fst \  # optional
  --g2p_en_sym g2p_en/phones.sym \  # optional
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
