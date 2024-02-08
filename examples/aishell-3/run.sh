#!/usr/bin/env bash

# Copyright 2022 Jie Chen
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="0,1,2,3"

stage=0  # start from -1 if you need to download data
stop_stage=3

dataset_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
dataset_dir=. # path to dataset directory

dir=exp/v1  # training dir
config=configs/v1.json

data=data
test_audio=test_audio

. tools/parse_options.sh || exit 1;


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $dataset_url $dataset_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare data for training/validation
  mkdir -p $data
  python tools/gen_pinyin_lexicon.py \
    --with-zero-initial --with-tone --with-r \
    $data/lexicon.txt \
    $data/phones.list
  python local/prepare_data.py \
    $data/lexicon.txt \
    $dataset_dir/data_aishell3 \
    $data/all.txt

  cat $data/all.txt | awk -F '|' '{print $2}' | \
    sort | uniq | awk '{print $0, NR-1}' > $data/speaker.txt
  echo 'sil 0' > $data/phones.txt
  cat $data/all.txt | awk -F '|' '{print $3}' | \
    awk '{for (i=1;i<=NF;i++) print $i}' | sort | uniq | \
    grep -v 'sil' | awk '{print $0, NR}' >> $data/phones.txt

  # Split train/validation
  shuf --random-source=<(yes 777) $data/all.txt > $data/train.txt
  head -n 100 $data/train.txt > $data/val.txt
  sed -i '1,100d' $data/train.txt
  head -n 10 $data/train.txt > $data/test.txt
  sed -i '1,10d' $data/train.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    vits/train.py -c $config -m $dir \
      --train_data $data/train.txt \
      --val_data $data/val.txt \
      --speaker_table $data/speaker.txt \
      --phone_table $data/phones.txt \
      --num_workers 8
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  mkdir -p $test_audio
  python vits/inference.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --checkpoint $dir/G_90000.pth \
    --test_file $data/test.txt \
    --outdir $test_audio
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  mkdir -p $test_audio
  python vits/export_onnx.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --checkpoint $dir/G_90000.pth \
    --onnx_model $dir/G_90000.onnx

  python vits/inference_onnx.py --cfg $config \
    --speaker_table $data/speaker.txt \
    --phone_table $data/phones.txt \
    --onnx_model $dir/G_90000.onnx \
    --test_file $data/test.txt \
    --outdir $test_audio
fi
