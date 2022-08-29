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

### Polyphone

| system         | ACC    |
|----------------|--------|
| BERT-polyphone | 0.9778 |
| BERT-MLT       | 0.9797 |


### Prosody

| system       | PW-F1  | PPH-F1 | IPH-F1 |
|--------------|--------|--------|--------|
| BERT-prosody | 0.9308 | 0.8058 | 0.8596 |
| BERT-MLT     | 0.9334 | 0.8088 | 0.8559 |
