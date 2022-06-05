#!/usr/bin/env python
# Copyright 2022 Binbin Zhang(binbzha@qq.com), Jie Chen(unrea1sama@outlook.com)
"""Generate lab files from data list for alignment
"""

import argparse
import pathlib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=str, help='Path to wav.txt.')
    parser.add_argument("speaker", type=str, help='Path to speaker.txt.')
    parser.add_argument(
        "text",
        type=str,
        help=('Path to text.txt. ',
              'It should only contain phonemes and special tokens.'))
    parser.add_argument('special_tokens',
                        type=str,
                        help='Path to special_token.txt.')
    parser.add_argument(
        'pronounciation_dict',
        type=str,
        help='Path to export pronounciation dictionary for MFA.')
    parser.add_argument('output_dir',
                        type=str,
                        help='Path to directory for exporting .lab files.')
    return parser.parse_args()


def main(args):
    output_dir = pathlib.Path(args.output_dir)
    pronounciation_dict = set()
    with open(args.special_tokens) as fin:
        special_tokens = set([x.strip() for x in fin.readlines()])

    with open(args.wav) as fwav, open(args.speaker) as fspeaker, open(
            args.text) as ftext:
        for wav_path, speaker, text in zip(fwav, fspeaker, ftext):
            wav_path, speaker, text = (pathlib.Path(wav_path.strip()),
                                       speaker.strip(), text.strip().split())
            lab_dir = output_dir / speaker
            lab_file = output_dir / speaker / '{}.lab'.format(wav_path.stem)

            lab_dir.mkdir(parents=True, exist_ok=True)
            with lab_file.open('w') as fout:
                text_no_special_tokens = list(
                    filter(lambda x: x not in special_tokens, text))
                pronounciation_dict |= set(text_no_special_tokens)
                fout.writelines([' '.join(text_no_special_tokens)])
    with open(args.pronounciation_dict, 'w') as fout:
        fout.writelines([
            '{} {}\n'.format(symbol, symbol) for symbol in pronounciation_dict
        ])


if __name__ == '__main__':
    main(get_args())
