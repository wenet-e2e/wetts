# Copyright 2022 Zhengxi Liu (xcmyz@outlook.com)

import os
import argparse
import jsonlines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', "-d", type=str, help='Path to BZNSYP dataset.')
    parser.add_argument('--save_path', "-s", type=str, help='Path to save processed data.')
    parser.add_argument('--val_samples', "-v", type=int, default=20, help='Number of validation samples.')
    parser.add_argument('--test_samples', "-t", type=int, default=20, help='Number of test samples.')
    return parser.parse_args()


def parse_interval(file_name):
    info_dict = dict()
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        sentence_id = os.path.split(file_name)[1].split(".")[0]
        duration = float(lines[4].replace("\n", ""))
        alignment = lines[12:]
        length_alignment = len(alignment)
        alignment_info = list()
        for i in range(length_alignment // 3):
            phone_alignment = (alignment[i * 3 + 2][1:-2],
                               (float(alignment[i * 3 + 0][:-1]),
                                float(alignment[i * 3 + 1][:-1])))
            alignment_info.append(phone_alignment)
        info_dict.update({"sentence_id": sentence_id,
                          "duration": duration,
                          "alignment": alignment_info})
    return info_dict


def generate_duration(interval_info):
    phone_list = []
    duration_list = []
    for ele in interval_info["alignment"]:
        phone_list.append(ele[0])
        duration_list.append(ele[1][1])
    return phone_list, duration_list


def generate_data_list(args):
    path = os.path.join(args.data_path, "PhoneLabeling")
    filename_list = os.listdir(path)
    filename_list.sort()
    data_list = []
    for filename in filename_list:
        info_dict = parse_interval(os.path.join(path, filename))
        key = info_dict['sentence_id']
        relative_path = os.path.join(args.data_path, "Wave", f"{info_dict['sentence_id']}.wav")
        if os.path.exists(relative_path):
            wav = os.path.abspath(relative_path)
            phone_list, duration_list = generate_duration(info_dict)
            data_list.append({
                'key': key,
                'wav_path': wav,
                'speaker': 1,
                'text': phone_list,
                'duration': [float(d) for d in duration_list]
            })
    train_data_list = data_list[args.val_samples + args.test_samples:]
    val_data_list = data_list[:args.val_samples]
    test_data_list = data_list[args.val_samples:args.val_samples + args.test_samples]
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "test"), exist_ok=True)
    with jsonlines.open(os.path.join(args.save_path, "train", "datalist.jsonl"), 'w') as fdatalist:
        for data in train_data_list:
            fdatalist.write(data)
    with jsonlines.open(os.path.join(args.save_path, "val", "datalist.jsonl"), 'w') as fdatalist:
        for data in val_data_list:
            fdatalist.write(data)
    with jsonlines.open(os.path.join(args.save_path, "test", "datalist.jsonl"), 'w') as fdatalist:
        for data in test_data_list:
            fdatalist.write(data)


def main(args):
    generate_data_list(args)


if __name__ == "__main__":
    main(get_args())
