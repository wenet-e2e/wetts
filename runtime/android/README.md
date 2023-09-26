# Usage

Most of AI engineers are not familiar with Android development, this is a simple ‘how to’.

1. Train your model with your data

2. Export pytorch model to onnx model

3. Convert onnx model for mobile deployment

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort your-model.onnx
```

you will get `your-model.ort` and `your-model.with_runtime_opt.ort`

``` bash
$ tree app/src/main/assets
app/src/main/assets
├── frontend
│   ├── final.ort
│   ├── frontend.flags
│   ├── g2p_en
│   │   ├── README.md
│   │   ├── cmudict.dict
│   │   ├── model.fst
│   │   └── phones.sym
│   ├── lexicon
│   │   ├── lexicon.txt
│   │   ├── pinyin_dict.txt
│   │   ├── polyphone.txt
│   │   ├── polyphone_phone.txt
│   │   └── prosody.txt
│   ├── tn
│   │   ├── zh_tn_tagger.fst
│   │   └── zh_tn_verbalizer.fst
│   └── vocab.txt
└── vits
    ├── final.ort
    ├── phones.txt
    ├── speaker.txt
    └── vits.flags

$ head app/src/main/assets/frontend/frontend.flags
--tagger=frontend/tn/zh_tn_tagger.fst
--verbalizer=frontend/tn/zh_tn_verbalizer.fst
--cmudict=frontend/g2p_en/cmudict.dict
--g2p_en_model=frontend/g2p_en/model.fst
--g2p_en_sym=frontend/g2p_en/phones.sym
--char2pinyin=frontend/lexicon/pinyin_dict.txt
--pinyin2id=frontend/lexicon/polyphone.txt
--pinyin2phones=frontend/lexicon/lexicon.txt
--vocab=frontend/vocab.txt
--g2p_prosody_model=frontend/final.ort

$ cat app/src/main/assets/vits/vits.flags
--sampling_rate=16000
--speaker2id=vits/speaker.txt
--phone2id=vits/phones.txt
--vits_model=vits/final.ort
```

4. Install Android Studio and open path of wetts/runtime/android and build

5. Install `app/build/outputs/apk/debug/app-debug.apk` to your phone and try it.
