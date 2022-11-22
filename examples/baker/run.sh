#!/usr/bin/env bash

# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"  # specify your gpu id for training

stage=0  # start from -1 if you need to download data
stop_stage=3

config=configs/base.json  #
dir=exp/base  # training dir
test_audio=test_audio

# Please download data from https://www.data-baker.com/data/index/TNtts/, and
# set `raw_data_dir` to your data.
raw_data_dir=/mnt/mnt-data-1/binbin.zhang/data/BZNSYP
data=data
use_onnx=false

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare data for training/validation
  mkdir -p $data
  python tools/gen_pinyin_lexicon.py --with-tone --with-r \
      $data/lexicon.txt $data/phones.list
  python local/prepare_data.py $data/lexicon.txt \
      $raw_data_dir/ProsodyLabeling/000001-010000.txt \
      $raw_data_dir/Wave > $data/all.txt
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
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
  python vits/train.py -c $config -m $dir \
    --train_data $data/train.txt \
    --val_data $data/val.txt \
    --phone_table $data/phones.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python vits/export_onnx.py  \
    --checkpoint $dir/G_90000.pth \
    --cfg configs/base.json \
    --onnx_model $dir/G_90000.onnx \
    --phone data/phones.txt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  [ ! -d ${test_audio} ] && mkdir ${test_audio}
  if $use_onnx; then
    python vits/inference_onnx.py  \
      --onnx_model $dir/G_90000.onnx --cfg $config \
      --outdir $test_audio \
      --phone $data/phones.txt \
      --test_file $data/test.txt
  else
    python vits/inference.py  \
      --checkpoint $dir/G_90000.pth --cfg $config \
      --outdir $test_audio \
      --phone $data/phones.txt \
      --test_file $data/test.txt
  fi
fi
