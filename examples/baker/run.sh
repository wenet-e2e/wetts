#!/usr/bin/env bash

# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

stage=0  # start from -1 if you need to download data
stop_stage=2

config=configs/base.json  #
dir=exp/base  # training dir
test_audio=test_audio

# Please download data from https://www.data-baker.com/data/index/TNtts/, and
# set `raw_data_dir` to your data.
raw_data_dir=/mnt/mnt-data-1/binbin.zhang/data/BZNSYP
data=data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare data for training/validation
  mkdir -p $data
  python tools/gen_pinyin_lexicon.py --with-tone --with-r \
      $data/lexicon.txt $data/phones.list
  python local/prepare_data.py $data/lexicon.txt \
      $raw_data_dir/script/000001-010000.txt \
      $raw_data_dir/wave_48k/000001-010000 > $data/all.txt
  # phone with 0 is kept for <blank>
  cat $data/all.txt | awk -F '\|' '{print $2}' | \
      awk '{ for (i=1;i<=NF;i++) print $i}' | \
      sort | uniq | awk '{print $0, NR}' > $data/phones.txt
  # Split train/validation
  cat $data/all.txt | shuf --random-source=<(yes 777) | head -n 110 | \
      awk -F '\|' '{print $1}' > $data/val.key
  cat $data/all.txt | grep -f $data/val.key > $data/val.txt
  head -10 $data/val.txt > $data/test.txt
  sed -i '1,10d' $data/val.txt
  cat $data/all.txt | grep -v -f $data/val.key > $data/train.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python vits/train.py -c $config -m $dir \
    --train_data $data/trian.txt \
    --val_data $data/val.txt \
    --phone_table $data/phones.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  [ ! -d ${test_audio} ] && mkdir ${test_audio}
  python vits/inference.py  \
    --checkpoint ./logs/exp/base/G_94000.pth --cfg $config \
    --outdir $test_audio \
    --phone $data/phones.txt \
    --test_file $data/test.txt
fi
