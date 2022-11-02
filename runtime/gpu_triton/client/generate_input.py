import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        default=None,
        help='a text file'
    )
    FLAGS = parser.parse_args()

    data = {"data": []}
    with open(FLAGS.text, "r", encoding="utf-8")as f:
        for line in f:
            audio_name, audio_text = line.strip().split("|", 1)
            li = {"text": [audio_text.strip('\n')]}
            data["data"].append(li)
    json.dump(data, open("input.json", "w", encoding="utf-8"), ensure_ascii=False)
