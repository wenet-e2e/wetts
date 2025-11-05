## Model Method

Please see [doc](../../wetts/frontend/README.md) for details.

## Data Description

Here are the details of the prosody and polyphone data used in the recipe.
The data are either collected from web or contributed by the community.


### Polyphone

| corpus | number | source or contributors             |
|--------|--------|------------------------------------|
| g2pM   | 100000 | https://github.com/kakaobrain/g2pM |
|        |        |                                    |

TODO(Binbin Zhang): Add more data


### Prosody

| corpus  | number | source or contributors                      |
|---------|--------|---------------------------------------------|
| biaobei | 10000  | https://www.data-baker.com/open_source.html |
|         |        |                                             |

TODO(Binbin Zhang): Add more data

## Benchmark

BERT-MLT is for polyphone and prosody joint training.

Bert base model: https://huggingface.co/google-bert/bert-base-chinese
TinyBert base model: https://huggingface.co/JeremiahZ/TinyBERT_4L_zh_backup/tree/main

### Polyphone

| system         | ACC    |
|----------------|--------|
| BERT-polyphone | 0.9778 |
| BERT-MLT       | 0.9797 |
| TinyBert-4L-MLT| 0.9736 |


### Prosody

| system                    | PW-F1  | PPH-F1 | IPH-F1 |
|---------------------------|--------|--------|--------|
| BERT-prosody              | 0.9308 | 0.8058 | 0.8596 |
| BERT-MLT                  | 0.9334 | 0.8088 | 0.8559 |
| TinyBert-4L-MLT           | 0.9241 | 0.7769 | 0.7829 |
| BERT-prosody (exclude #4) | 0.9233 | 0.7074 | 0.6120 |
| BERT-MLT (exclude #4)     | 0.9261 | 0.7146 | 0.6140 |
