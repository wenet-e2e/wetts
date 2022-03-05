#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)

stage=0 # start from -1 if you need to download data
stop_stage=5

data_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
data_dir=/mnt/mnt-data-1/binbin.zhang/data
data=data
dir=exp

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $data_url $data_dir
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $data/train $data/test $data/dict
  # Prepare alignment lab for MFA tools
  local/prepare_align_lab.py $data_dir/data_aishell3/train $data/train
  python tools/generate_mfa_pinyin_lexicon.py --with-tone --with-r \
      $data/dict/lexicon.txt $data/dict/phones.txt
  # MFA alignment
  conda_base=$(conda info --base)
  source $conda_base/bin/activate aligner
  mfa train -j 32 --phone_set PINYIN --overwrite \
      -a $data_dir/data_aishell3/train/wav \
      $data/train/lab $data/dict/lexicon.txt \
      $dir/align_model.zip $data/train/TextGrid
  conda deactivate
fi

