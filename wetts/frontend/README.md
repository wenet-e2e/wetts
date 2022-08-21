# WeTTS Frontend System

## Keynotes

Motivated by [Unified Mandarin TTS Front-end Based on Distilled BERT Model](https://arxiv.org/pdf/2012.15404.pdf),
we want to give a simple, production ready, and unified frontend solution in `wetts`.


## Roadmap

- [x] Server prosody and polyphone based on BERT.
- [ ] On-device prosody and polyphone solution.
- [ ] Joint training with word break and POS to further improve performance(Optional).
- [ ] Text normalization solution.

## Data Format

### Prosody

The prosody format is like following, `#n` is prosody rank.

```
蔡少芬 #2 拍拖 #2 也不认啦 #4
瓦塔拉 #1 总统 #1 已 #1 下令 #3 坚决 #1 回应 #1 袭击者 #4
```

### Polyphone

The polyphone is surrounded with `▁` in training corpus.


```
宋代出现了▁le5▁燕乐音阶的记载
2011年9月17日，爆发了▁le5▁占领华尔街示威活动
```

