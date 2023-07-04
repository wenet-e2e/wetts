import os
import sys
import random

if len(sys.argv) != 7:
    print('Usage: prepare_data.py lexicon in_data_dir all_data_path '
          'test_data_path valid_data_path train_data_path')
    sys.exit(-1)

lexicon = {}
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.strip().split()
        lexicon[arr[0]] = arr[1:]

cnt = 0
with open(sys.argv[2], encoding='utf8') as fin, \
     open(sys.argv[3], 'w', encoding='utf8') as fout_all, \
     open(sys.argv[4], 'w', encoding='utf8') as fout_test, \
     open(sys.argv[5], 'w', encoding='utf8') as fout_valid, \
     open(sys.argv[6], 'w', encoding='utf8') as fout_train:

    lines = [x.strip() for x in fin.readlines()]
    random.shuffle(lines)

    for line in lines:

        speaker, duration, text, pinyin_list, wav_path = line.split(' ')

        if speaker == "spk":  # 跳过首行
            continue

        # 跳过含英文的case
        skip_cases = ["UP", "B", "O", "live"]
        skip = False
        for case in skip_cases:
            if case in text:
                skip = True
        if skip is True:
            continue

        phones = ["sil"]
        for x in pinyin_list.split(','):
            if x in ["#0", "#1", "#2", "#3"]:
                phones.append(x)
            elif x in lexicon:
                phones.extend(lexicon[x])
            elif x == "n2":
                phones.extend(lexicon["en2"])
            else:
                print('{} \n{} \nOOV {}'.format(text, pinyin_list, x))
                sys.exit(-1)
        phones.append("sil")

        write_line = '{}|{}|{}\n'.format(wav_path, speaker, ' '.join(phones))

        fout_all.write(write_line)
        if cnt < 10:
            fout_test.write(write_line)
        elif 10 <= cnt < 110:
            fout_valid.write(write_line)
        else:
            fout_train.write(write_line)
        cnt += 1
