""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from g2p_en import G2p
import nltk
from unidecode import unidecode


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


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def filter(phonemes, use_prosody=False):
    if not use_prosody:
        return [phoneme for phoneme in phonemes if not re.match("^[-,!?.' ]+$", phoneme)]

    phones = []
    for phoneme in phonemes:
        if phoneme == " " and len(phones) > 0 and "#" not in phones[-1]:
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


def english_cleaners(text, use_prosody=False):
    """Pipeline for English text, including abbreviation expansion."""
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = g2p(text)
    phonemes = filter(phonemes, use_prosody)
    return phonemes
