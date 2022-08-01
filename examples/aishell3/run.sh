#!/usr/bin/env bash
# Copyright 2022 Binbin Zhang(binbzha@qq.com), Jie Chen
. path.sh

stage=                            # start from -1 if you need to download data
stop_stage=

dataset_url=https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz
dataset_dir=                          # path to dataset directory

fastspeech2_outputdir=fastspeech2
fastspeech2_config=conf/fastspeech2.yaml

hifigan_outputdir=hifigan
hifigan_config=conf/hifigan_v1.yaml

conda_base=$(conda info --base)
source $conda_base/bin/activate wetts

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # Download data
  local/download_data.sh $data_url $dataset_dir
fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Prepare wav.txt, speaker.txt and text.txt
  python local/prepare_data_list.py $dataset_dir $fastspeech2_outputdir/wav.txt \
    $fastspeech2_outputdir/speaker.txt $fastspeech2_outputdir/text.txt
  # Prepare special_tokens. Special tokens in AISHELL3 are % and $.
  (echo %; echo $;) > $fastspeech2_outputdir/special_token.txt
  # Prepare lexicon.
  python tools/gen_mfa_pinyin_lexicon.py --with-tone --with-r \
    $fastspeech2_outputdir/lexicon.txt $fastspeech2_outputdir/phone.txt
  # Convert text in text.txt to phonemes.
  python local/convert_text_to_phn.py $fastspeech2_outputdir/text.txt \
    $fastspeech2_outputdir/lexicon.txt $fastspeech2_outputdir/special_token.txt \
    $fastspeech2_outputdir/text.txt
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Prepare alignment lab and pronounciation dictionary for MFA tools
  python local/prepare_alignment.py $fastspeech2_outputdir/wav.txt \
    $fastspeech2_outputdir/speaker.txt $fastspeech2_outputdir/text.txt \
    $fastspeech2_outputdir/special_token.txt \
    $fastspeech2_outputdir/mfa_pronounciation_dict.txt \
    $fastspeech2_outputdir/lab/
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # MFA alignment
  mfa train -j 32 --phone_set PINYIN --overwrite \
      -a $dataset_dir/train/wav \
      $fastspeech2_outputdir/lab \
      $fastspeech2_outputdir/mfa_pronounciation_dict.txt \
      $fastspeech2_outputdir/mfa/mfa_model.zip $fastspeech2_outputdir/TextGrid
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  python tools/gen_alignment_from_textgrid.py \
    $fastspeech2_outputdir/wav.txt \
    $fastspeech2_outputdir/speaker.txt $fastspeech2_outputdir/text.txt \
    $fastspeech2_outputdir/special_token.txt $fastspeech2_outputdir/TextGrid \
    $fastspeech2_outputdir/aligned_wav.txt \
    $fastspeech2_outputdir/aligned_speaker.txt \
    $fastspeech2_outputdir/duration.txt \
    $fastspeech2_outputdir/aligned_text.txt
  # speaker to id map
  cat $fastspeech2_outputdir/aligned_speaker.txt | awk '{print $1}' | sort | uniq | \
      awk '{print $1, NR-1}' > $fastspeech2_outputdir/spk2id
  # phone to id map
  (echo "<pad> 0"; echo "<eos> 1"; echo "<unk> 2") > \
    $fastspeech2_outputdir/phn2id
  cat $fastspeech2_outputdir/aligned_text.txt | \
    awk '{for (i=1;i<NF;i++) print $i}' | sort | uniq | awk '{print $1, NR+2}' \
    >> $fastspeech2_outputdir/phn2id
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # generate training, validation and test samples
  python local/train_val_test_split.py $fastspeech2_outputdir/aligned_wav.txt \
  $fastspeech2_outputdir/aligned_speaker.txt \
  $fastspeech2_outputdir/aligned_text.txt \
  $fastspeech2_outputdir/duration.txt $fastspeech2_outputdir
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Prepare training samples
  python local/make_data_list.py $fastspeech2_outputdir/train/train_wav.txt \
      $fastspeech2_outputdir/train/train_speaker.txt \
      $fastspeech2_outputdir/train/train_text.txt \
      $fastspeech2_outputdir/train/train_duration.txt \
      $fastspeech2_outputdir/train/datalist.jsonl
  # Prepare validation samples
  python local/make_data_list.py $fastspeech2_outputdir/val/val_wav.txt \
      $fastspeech2_outputdir/val/val_speaker.txt \
      $fastspeech2_outputdir/val/val_text.txt \
      $fastspeech2_outputdir/val/val_duration.txt \
      $fastspeech2_outputdir/val/datalist.jsonl
  # Prepare test samples
  python local/make_data_list.py $fastspeech2_outputdir/test/test_wav.txt \
      $fastspeech2_outputdir/test/test_speaker.txt \
      $fastspeech2_outputdir/test/test_text.txt \
      $fastspeech2_outputdir/test/test_duration.txt \
      $fastspeech2_outputdir/test/datalist.jsonl
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Compute mel, f0, energy CMVN
  total=$(cat $fastspeech2_outputdir/train/datalist.jsonl | wc -l)
  python wetts/bin/compute_cmvn.py --total $total --num_workers 32 \
      $fastspeech2_config $fastspeech2_outputdir/train/datalist.jsonl \
      $fastspeech2_outputdir/train
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  # train fastspeech2
  EPOCH=
  python wetts/bin/train.py fastspeech2 --num_workers 32 \
      --config $fastspeech2_config \
      --train_data_list $fastspeech2_outputdir/train/datalist.jsonl \
      --val_data_list $fastspeech2_outputdir/val/datalist.jsonl \
      --cmvn_dir $fastspeech2_outputdir/train \
      --spk2id_file $fastspeech2_outputdir/spk2id \
      --phn2id_file $fastspeech2_outputdir/phn2id \
      --special_tokens_file $fastspeech2_outputdir/special_token.txt \
      --log_dir log/ \
      --batch_size 64 \
      --epoch $EPOCH
fi

FASTSPEECH2_INFERENCE_OUTPUTDIR=$fastspeech2_outputdir/inference_mels # path to directory for inferenced mels
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  # inference fastspeech2
  TEXT_FILE=test_samples.txt                 # path to text file, each line contains one sample for inference
  SPEAKER_FILE=test_samples_speakers.txt     # path to speaker file, each line contains one speaker name for corresponding line in text file
  FASTSPEECH2_CKPT_PATH=                     # path to fastspeech2 checkpoint file
  python wetts/bin/inference.py fastspeech2 \
      --num_workers 4 \
      --batch_size 64 \
      --config $fastspeech2_config \
      --text_file $TEXT_FILE \
      --speaker_file $SPEAKER_FILE \
      --lexicon_file $fastspeech2_outputdir/lexicon.txt \
      --cmvn_dir $fastspeech2_outputdir/train \
      --spk2id_file $fastspeech2_outputdir/spk2id \
      --phn2id_file $fastspeech2_outputdir/phn2id \
      --special_token_file $fastspeech2_outputdir/special_token.txt \
      --export_dir $FASTSPEECH2_INFERENCE_OUTPUTDIR \
      --ckpt $FASTSPEECH2_CKPT_PATH
fi


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  # train hifigan using fastspeech2 training dataset
  EPOCH=                              # Number of epoch for training hifigan
  python wetts/bin/hifigan_train.py train \
      --num_workers 32 \
      --batch_size_hifigan 32 \
      --fastspeech2_train_datalist $fastspeech2_outputdir/train/datalist.jsonl \
      --fastspeech2_val_datalist $fastspeech2_outputdir/val/datalist.jsonl \
      --hifigan_config $hifigan_config \
      --epoch $EPOCH \
      --export_dir $hifigan_outputdir/train
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  # finetune hifigan using fastspeech2 training dataset
  FASTSPEECH2_CKPT_PATH=          # path to fastspeech2 checkpoint
  HIFIGAN_CKPT_PATH=              # path to hifigan generator and discriminator checkpoint
                                  # e.g. $HIFIGAN_CKPT_PATH='g_02500000 do_02500000'
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan
  # This stage will continue training hifigan using checkpoints specified with --hifigan_ckpt.
  # Mels predicted by fastspeech2 and ground truth audios from fastspeech2 training dataset are
  # used as training dataset for hifigan.
  # Commenting out --hifigan_ckpt will train hifigan from scratch.
  # When running finetune command for the first time, --generate_samples should be specified.
  EPOCH=                          # number of epoch for finetune
  python wetts/bin/hifigan_train.py finetune \
      --num_workers 32 \
      --batch_size_hifigan 32 \
      --batch_size_fastspeech2 32 \
      --fastspeech2_config conf/fastspeech2.yaml \
      --fastspeech2_train_datalist $fastspeech2_outputdir/train/datalist.jsonl \
      --hifigan_config $hifigan_config \
      --phn2id_file $fastspeech2_outputdir/phn2id \
      --spk2id_file $fastspeech2_outputdir/spk2id \
      --special_tokens_file $fastspeech2_outputdir/special_token.txt \
      --cmvn_dir $fastspeech2_outputdir/train \
      --fastspeech2_ckpt $FASTSPEECH2_CKPT_PATH \
      --hifigan_ckpt $HIFIGAN_CKPT_PATH \
      --epoch $EPOCH \
      --export_dir $hifigan_outputdir/finetune \
      --generate_samples
fi


if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  # hifigan inference
  HIFIGAN_GENERATOR_CKPT_PATH=    # path to hifigan generator checkpoint
                                  # e.g. $HIFIGAN_GENERATOR_CKPT_PATH=g_02500000
                                  # pretrained hifigan checkpoint can be obtained from:
                                  # https://github.com/jik876/hifi-gan

  python wetts/bin/hifigan_inference.py \
      --num_workers 4 \
      --batch_size 32 \
      --config $hifigan_config \
      --datalist $FASTSPEECH2_INFERENCE_OUTPUTDIR/fastspeech2_mel_prediction.jsonl \
      --ckpt $HIFIGAN_GENERATOR_CKPT_PATH \
      --export_dir $hifigan_outputdir/wavs
fi
