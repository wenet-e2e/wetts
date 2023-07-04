#!/usr/bin/env bash
[ -f path.sh ] && . path.sh

export CUDA_VISIBLE_DEVICES="0"

stage=1
stop_stage=2

config="configs/base.json"
data="data/yuanshen"
exp_dir="exp/base"

checkpoint="${exp_dir}/G_90000.pth"
test_output="test_output"

tools="../../tools"
. ${tools}/parse_options.sh || exit 1;

# stage 0: make lexicon and phones.
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p ${data}
  python ${tools}/gen_pinyin_lexicon.py \
    --with-tone \
    --with-r \
    "${data}/lexicon.list" \
    "${data}/phones.list"
fi

# stage 1: load ori data and make train.txt val.txt and test.txt
ori_label_file="G:\Yuanshen\2.jiaba_cut_label.txt"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # test 10 pieces, valid 100 pieces
  python local/prepare_data.py \
    "${data}/lexicon.list" \
    "${ori_label_file}" \
    "${data}/all.txt" \
    "${data}/test.txt" \
    "${data}/val.txt" \
    "${data}/train.txt"
fi

# stage 2: make phones.txt
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # phone with 0 is kept for <blank>
  cat "${data}/all.txt" | awk -F '|' '{print $3}' | \
    awk '{ for (i=1;i<=NF;i++) print $i}' | \
    sort | uniq | awk '{print $0, NR}' \
    > "${data}/phones.txt"
  cat "${data}/all.txt" | awk -F '|' '{print $2}' | \
    sort | uniq | awk '{print $0, NR}' > "${data}/speaker.txt"
fi

# stage 3: train
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
  python vits/train.py \
    -c ${config} \
    -m ${exp_dir} \
    --train_data     "${data}/train.txt" \
    --val_data       "${data}/val.txt" \
    --phone_table    "${data}/phones.txt" \
    --speaker_table  "${data}/speaker.txt"
fi

# stage 4: test
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  [ ! -d ${test_output} ] && mkdir ${test_output}
  python vits/inference.py  \
    --checkpoint     ${checkpoint} \
    --cfg            ${config} \
    --outdir         ${test_output} \
    --phone_table    "${data}/phones.txt" \
    --test_file      "${data}/test.txt" \
    --speaker_table  "${data}/speaker.txt"
fi
