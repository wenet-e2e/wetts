# Copyright 2022 Zhengxi Liu (xcmyz@outlook.com)

import os
import argparse
import jsonlines


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', "-d", type=str, help='Path to BZNSYP dataset.')
    parser.add_argument('--save_path', "-s", type=str, help='Path to save processed data.')
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


def generate_data_list(data_path, save_path):
    path = os.path.join(data_path, "PhoneLabeling")
    filename_list = os.listdir(path)
    filename_list.sort()
    with jsonlines.open(save_path, 'w') as fdatalist:
        for filename in filename_list:
            info_dict = parse_interval(os.path.join(path, filename))
            key = info_dict['sentence_id']
            relative_path = os.path.join(data_path, "Wave", f"{info_dict['sentence_id']}.wav")
            if os.path.exists(relative_path):
                wav = os.path.abspath(relative_path)
                phone_list, duration_list = generate_duration(info_dict)
                fdatalist.write({
                    'key': key,
                    'wav_path': wav,
                    'speaker': 1,
                    'text': phone_list,
                    'duration': [float(d) for d in duration_list]
                })


def main():
    args = get_args()
    generate_data_list(args.data_path, args.save_path)


if __name__ == "__main__":
    main()
