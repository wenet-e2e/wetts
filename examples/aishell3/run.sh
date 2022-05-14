#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)
. path.sh

stage=0 # start from -1 if you need to download data
stop_stage=0

data_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
data_dir=/mnt/mnt-data-1/binbin.zhang/data

config=conf/default.yaml
data=data
dump=dump
dir=exp

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $data_url $data_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $data/train $data/test $data/dict
  # Prepare alignment lab for MFA tools
  local/prepare_align_lab.py $data_dir/data_aishell3/train $data/train
  python tools/gen_mfa_pinyin_lexicon.py --with-tone --with-r \
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


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  mkdir -p $dump/train
  # Generate phone alignment information from textgrid, including
  # phone, duration, speaker
  python tools/gen_alignment_from_textgrid.py \
      --inputdir=$data/train/TextGrid \
      --outputdir=$dump/train \
      --config=$config
  # speaker to id map
  cat $dump/train/utt2spk | awk '{print $2}' | sort | uniq | \
      awk '{print $1, NR-1}' > $dump/spk2id
  # phone to id map
  (echo "<eos> 0"; echo "<unk> 1") > $dump/phn2id
  cat $dump/train/utt2phn | awk '{for (i=2;i<NF;i++) print $i}' | sort | \
      uniq | awk '{print $1, NR+1}' >> $dump/phn2id

  # Prepare training shards
  cp data/train/wav.scp $dump/train
  python tools/make_shard_list.py --num_utts_per_shard 1000 --shuffle \
      --num_threads 16 \
      $dump/train/wav.scp \
      $dump/train/utt2dur \
      $dump/train/utt2phn \
      $dump/train/utt2spk \
      $dump/train/shards \
      $dump/train/data.list

  # Compute mel, f0, energy CMVN
  total=$(cat $dump/train/utt2dur | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 32 \
      $config $dump/train/data.list $dump/train
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  python wetts/bin/train.py --num_workers 32 \
      --config $config \
      --train_list $dump/train/data.list \
      --cmvn_dir $dump/train \
      --spk2id_file $dump/spk2id \
      --phn2id_file $dump/phn2id
fi
