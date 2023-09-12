""" from https://github.com/keithito/tacotron """

import re
from g2p_en import G2p


g2p = G2p()

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_prosodies = ["#0", "#1", "#2", "#3", "#4"]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def filter(phonemes, use_prosody):
    phones = []
    if not use_prosody:
        for phoneme in phonemes:
            is_symbol = re.match("^[-,!?.' ]+$", phoneme)
            if not is_symbol:
                phones.append(phoneme)
        return phones

    for phoneme in phonemes:
        if re.match("^[']+$", phoneme):
            continue
        elif re.match("^[- ]+$", phoneme):
            if len(phones) > 0 and "#" not in phones[-1]:
                phones.append(_prosodies[1])
        elif re.match("^[,!?.]+$", phoneme):
            if len(phones) > 0 and "#" in phones[-1]:
                phones[-1] = max(phones[-1], _prosodies[3])
            else:
                phones.append(_prosodies[3])
        else:
            phones.append(phoneme)
    if "#" in phones[-1]:
        phones[-1] = _prosodies[-1]
    else:
        phones.append(_prosodies[-1])
    return phones


def english_cleaners(text, use_prosody):
    """Pipeline for English text, including abbreviation expansion."""
    text = text.lower()
    text = expand_abbreviations(text)
    phonemes = g2p(text)
    phonemes = filter(phonemes, use_prosody)
    return phonemes
