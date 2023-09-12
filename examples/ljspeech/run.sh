#!/usr/bin/env bash

# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh
export PYTHONPATH=.:$PYTHONPATH

export CUDA_VISIBLE_DEVICES="0,1,2,3"

stage=0  # start from -1 if you need to download data
stop_stage=3

dataset_url=https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
dataset_dir=. # path to dataset directory

config=configs/v3.json
dir=exp/v3  # training dir
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
  python local/prepare_data.py \
    $(realpath $dataset_dir)/LJSpeech-1.1 \
    $data/all.txt

  cat $data/all.txt | awk -F '|' '{print $2}' | \
    sort | uniq | awk '{print $0, NR}' > $data/speaker.txt

  echo 'sil 0' > $data/phones.txt
  cat $data/all.txt | awk -F '|' '{print $3}' | \
    awk '{for (i=1;i<=NF;i++) print $i}' | \
    sort | uniq | awk '{print $0, NR}' >> $data/phones.txt

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
  export MASTER_PORT=10086
  python vits/train.py -m $dir -c $config \
    --train_data $data/train.txt \
    --val_data $data/val.txt \
    --phone_table $data/phones.txt \
    --speaker_table $data/speaker.txt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python vits/export_onnx.py --cfg $config \
    --checkpoint $dir/G_90000.pth \
    --onnx_model $dir/G_90000.onnx \
    --phone_table $data/phones.txt \
    --speaker_table $data/speaker.txt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  [ ! -d ${test_audio} ] && mkdir ${test_audio}
  if $use_onnx; then
    python vits/inference_onnx.py --cfg $config \
      --onnx_model $dir/G_90000.onnx \
      --outdir $test_audio \
      --phone_table $data/phones.txt \
      --test_file $data/test.txt \
      --speaker_table $data/speaker.txt
  else
    python vits/inference.py --cfg $config \
      --checkpoint $dir/G_90000.pth \
      --outdir $test_audio \
      --phone_table $data/phones.txt \
      --test_file $data/test.txt \
      --speaker_table $data/speaker.txt
  fi
fi

