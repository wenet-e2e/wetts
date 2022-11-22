#!/usr/bin/env bash

# Copyright 2022 Jie Chen
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

stage=0  # start from -1 if you need to download data
stop_stage=1

dataset_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
dataset_dir=. # path to dataset directory

config=configs/base.json
dir=exp/base  # training dir
test_audio=test_audio

data=data
use_onnx=false

. tools/parse_options.sh || exit 1;


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $dataset_url $dataset_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare data for training/validation
  mkdir -p $data
  python tools/gen_pinyin_lexicon.py --with-tone --with-r \
      $data/lexicon.txt $data/phones.list
  python local/prepare_data.py $data/lexicon.txt \
      $dataset_dir/data_aishell3 $data/all.txt
  cat $data/all.txt | awk -F '|' '{print $2}' | \
      sort | uniq | awk '{print $0, NR}' > $data/speaker.txt
  # phone with 0 is kept for <blank>
  cat $data/all.txt | awk -F '|' '{print $3}' | \
      awk '{ for (i=1;i<=NF;i++) print $i}' | \
      sort | uniq | awk '{print $0, NR}' > $data/phones.txt
  # Split train/validation
  cat $data/all.txt | shuf --random-source=<(yes 777) | head -n 110 | \
      awk -F '|' '{print $1}' > $data/val.key
  cat $data/all.txt | grep -f $data/val.key > $data/val.txt
  head -10 $data/val.txt > $data/test.txt
  sed -i '1,10d' $data/val.txt
  cat $data/all.txt | grep -v -f $data/val.key > $data/train.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  export MASTER_ADDR=localhost
  export MASTER_PORT=10087
  python vits/train.py -c $config -m $dir \
    --train_data $data/train.txt \
    --val_data $data/val.txt \
    --phone_table $data/phones.txt \
    --speaker_table $data/speaker.txt
fi

