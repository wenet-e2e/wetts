import sys
import re

lexicon = {}

with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.strip().split()
        lexicon[arr[0]] = arr[1:]

with open(sys.argv[2], 'r', encoding='utf8') as fin:
    lines = fin.readlines()
    for i in range(0, len(lines), 2):
        key = lines[i][:6]
        content = lines[i][7:].strip()
        content = re.sub('[。，、“”？：……！（ ）—；]', '', content)
        if 'Ｐ' in content:  # ignore utt 002365
            continue
        chars = []
        prosody = {}

        j = 0
        while j < len(content):
            if content[j] == '#':
                prosody[len(chars) - 1] = content[j:j + 2]
                j += 2
            else:
                chars.append(content[j])
                j += 1
        if key == '005107':
            lines[i + 1] = lines[i + 1].replace(' ng1', ' en1')
        syllable = lines[i + 1].strip().split()
        s_index = 0
        phones = []
        for k, char in enumerate(chars):
            # 儿化音处理
            er_flag = False
            if char == '儿' and (s_index == len(syllable)
                                or syllable[s_index][0:2] != 'er'):
                er_flag = True
            else:
                phones.extend(lexicon[syllable[s_index]])
                s_index += 1
            if k in prosody:
                if er_flag:
                    phones[-1] = prosody[k]
                else:
                    phones.append(prosody[k])
            else:
                phones.append('#0')
        print('{}/{}.wav|sil {}'.format(sys.argv[3], key, ' '.join(phones)))
