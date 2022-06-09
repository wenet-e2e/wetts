#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com)
. path.sh

stage=0 # start from -1 if you need to download data
stop_stage=7

dataset_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
dataset_dir=~/AISHELL-3

outputdir=feature
config=conf/default.yaml

conda_base=$(conda info --base)
source $conda_base/bin/activate wetts

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $data_url $dataset_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare wav.txt, speaker.txt and text.txt
  python local/prepare_data_list.py $dataset_dir $outputdir/wav.txt \
    $outputdir/speaker.txt $outputdir/text.txt
  # Prepare special_tokens. Special tokens in AISHELL3 are % and $.
  (echo %; echo $;) > $outputdir/special_token.txt
  # Prepare lexicon.
  python tools/gen_mfa_pinyin_lexicon.py --with-tone --with-r \
    $outputdir/lexicon.txt $outputdir/phone.txt
  # Convert text in text.txt to phonemes.
  python local/convert_text_to_phn.py $outputdir/text.txt \
    $outputdir/lexicon.txt $outputdir/special_token.txt $outputdir/text.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Prepare alignment lab and pronounciation dictionary for MFA tools
  python local/prepare_alignment.py $outputdir/wav.txt $outputdir/speaker.txt \
    $outputdir/text.txt $outputdir/special_token.txt \
    $outputdir/mfa_pronounciation_dict.txt $outputdir/lab/
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # MFA alignment
  mfa train -j 32 --phone_set PINYIN --overwrite \
      -a $dataset_dir/train/wav \
      $outputdir/lab $outputdir/mfa_pronounciation_dict.txt \
      $outputdir/mfa/mfa_model.zip $outputdir/TextGrid
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  python tools/gen_alignment_from_textgrid.py $outputdir/wav.txt \
    $outputdir/speaker.txt $outputdir/text.txt $outputdir/special_token.txt \
    $outputdir/TextGrid $outputdir/aligned_wav.txt \
    $outputdir/aligned_speaker.txt $outputdir/duration.txt \
    $outputdir/aligned_text.txt
  # speaker to id map
  cat $outputdir/aligned_speaker.txt | awk '{print $1}' | sort | uniq | \
      awk '{print $1, NR-1}' > $outputdir/spk2id
  # phone to id map
  (echo "<pad> 0"; echo "<eos> 1"; echo "<unk> 2") > $outputdir/phn2id
  cat $outputdir/aligned_text.txt | awk '{for (i=1;i<NF;i++) print $i}' | \
    sort | uniq | awk '{print $1, NR+2}' >> $outputdir/phn2id
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # generate training, validation and test samples
  python local/train_val_test_split.py $outputdir/aligned_wav.txt \
  $outputdir/aligned_speaker.txt $outputdir/aligned_text.txt \
  $outputdir/duration.txt $outputdir
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Prepare training shards
  python tools/make_shard_list.py --num_utts_per_shard 1000 --shuffle \
      --num_threads 16 \
      $outputdir/train/train_wav.txt \
      $outputdir/train/train_speaker.txt \
      $outputdir/train/train_text.txt \
      $outputdir/train/train_duration.txt \
      $outputdir/train/shards \
      $outputdir/train/data.list
  # Prepare validation shards
  python tools/make_shard_list.py --num_utts_per_shard 1000 --shuffle \
      --num_threads 16 \
      $outputdir/val/val_wav.txt \
      $outputdir/val/val_speaker.txt \
      $outputdir/val/val_text.txt \
      $outputdir/val/val_duration.txt \
      $outputdir/val/shards \
      $outputdir/val/data.list
  # Prepare test shards
  python tools/make_shard_list.py --num_utts_per_shard 1000 --shuffle \
      --num_threads 16 \
      $outputdir/test/test_wav.txt \
      $outputdir/test/test_speaker.txt \
      $outputdir/test/test_text.txt \
      $outputdir/test/test_duration.txt \
      $outputdir/test/shards \
      $outputdir/test/data.list
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Compute mel, f0, energy CMVN
  total=$(cat $outputdir/train/train_wav.txt | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 32 \
      $config $outputdir/train/data.list $outputdir/train

  total=$(cat $outputdir/val/val_wav.txt | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 3 \
      $config $outputdir/val/data.list $outputdir/val

  total=$(cat $outputdir/test/test_wav.txt | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 3 \
      $config $outputdir/test/data.list $outputdir/test
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  python wetts/bin/train.py fastspeech2 --num_workers 32 \
      --config $config \
      --train_data_list $outputdir/train/data.list \
      --val_data_list $outputdir/val/data.list \
      --cmvn_dir $outputdir/train \
      --spk2id_file $outputdir/spk2id \
      --phn2id_file $outputdir/phn2id \
      --special_tokens_file $outputdir/special_token.txt
fi
