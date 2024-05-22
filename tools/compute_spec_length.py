#!/usr/bin/env python3
# author: @lsrami

import os
import sys
import json
from tqdm import tqdm
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def process_item(item):
    audiopath = item[0]
    src_sampling_rate = sf.info(audiopath).samplerate
    text = item[2]
    text = text.strip().split()
    if min_text_len <= len(text) and len(text) <= max_text_len:
        length = int(os.path.getsize(audiopath) * sampling_rate / src_sampling_rate) // (2 * hop_length)
        item.append(length)
        return item
    else:
        return None

def main(in_file, out_file):
    """
    Filter text & store spec lengths
    """

    audiopaths_sid_text = load_filepaths_and_text(in_file, split="|")

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_item, audiopaths_sid_text), total=len(audiopaths_sid_text)))

    # Filter out None results
    results = [result for result in results if result is not None]

    with open(out_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write("|".join([str(i) for i in item]) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <in_file> <config_file> <out_file>")
        sys.exit(1)
    in_file, config_file, out_file = sys.argv[1:4]

    with open(config_file, "r", encoding="utf8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = config["data"]

    min_text_len = hparams.get("min_text_len", 1)
    max_text_len = hparams.get("max_text_len", 190)
    sampling_rate = hparams.get("sampling_rate", 22050)
    hop_length = hparams.get("hop_length", 256)
    print(min_text_len, max_text_len, sampling_rate, hop_length)

    main(in_file, out_file)
