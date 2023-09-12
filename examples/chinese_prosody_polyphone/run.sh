#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

stage=0
stop_stage=4
url=https://wetts-1256283475.cos.ap-shanghai.myqcloud.com/data

dir=exp

. tools/parse_options.sh


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download prosody and polyphone
  mkdir -p data/download
  pushd data/download
  wget -c $url/polyphone.tar.gz && tar zxf polyphone.tar.gz
  wget -c $url/prosody.tar.gz && tar zxf prosody.tar.gz
  popd
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Combine prosody data
  mkdir -p data/prosody
  cat data/download/prosody/biaobei/train.txt > data/prosody/train.txt
  cat data/download/prosody/biaobei/cv.txt > data/prosody/cv.txt
  # Combine polyphone data
  mkdir -p data/polyphone
  cat data/download/polyphone/g2pM/train.txt > data/polyphone/train.txt
  cat data/download/polyphone/g2pM/dev.txt > data/polyphone/cv.txt
  cat data/download/polyphone/g2pM/test.txt > data/polyphone/test.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  mkdir -p $dir
  python frontend/train.py \
    --gpu 2 \
    --lr 0.001 \
    --num_epochs 10 \
    --batch_size 32 \
    --log_interval 10 \
    --polyphone_weight 0.1 \
    --polyphone_dict lexicon/polyphone.txt \
    --train_polyphone_data data/polyphone/train.txt \
    --cv_polyphone_data data/polyphone/cv.txt \
    --prosody_dict lexicon/prosody.txt \
    --train_prosody_data data/prosody/train.txt \
    --cv_prosody_data data/prosody/cv.txt \
    --model_dir $dir
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test polyphone, metric: accuracy
  python frontend/test_polyphone.py \
    --polyphone_dict lexicon/polyphone.txt \
    --prosody_dict lexicon/prosody.txt \
    --test_data data/polyphone/test.txt \
    --batch_size 32 \
    --checkpoint $dir/9.pt

  # Test prosody, metric: F1-score
  python frontend/test_prosody.py \
    --polyphone_dict lexicon/polyphone.txt \
    --prosody_dict lexicon/prosody.txt \
    --test_data data/prosody/cv.txt \
    --batch_size 32 \
    --checkpoint $dir/9.pt
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # export onnx model
  python frontend/export_onnx.py \
    --polyphone_dict lexicon/polyphone.txt \
    --prosody_dict lexicon/prosody.txt \
    --checkpoint $dir/9.pt \
    --onnx_model $dir/9.onnx
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # g2p
  # text: 八方财宝进
  # pinyin ['ba1', 'fang1', 'cai2', 'bao3', 'jin4']
  # prosody [0 1 0 0 4]
  python frontend/g2p_prosody.py \
    --text "八方财宝进" \
    --hanzi2pinyin_file lexicon/pinyin_dict.txt \
    --polyphone_file lexicon/polyphone.txt \
    --polyphone_prosody_model $dir/9.onnx
fi
