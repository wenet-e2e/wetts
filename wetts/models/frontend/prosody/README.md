# WeTTS Prosody System

## Step 1: Prepare prosody data

The prosody format is like following, `#n` is prosody rank.

```
蔡少芬 #2 拍拖 #2 也不认啦 #4
瓦塔拉 #1 总统 #1 已 #1 下令 #3 坚决 #1 回应 #1 袭击者 #4
我 #1 发了 #1 围脖 #3 要 #1 不要 #1 来 #1 转一下啊 #4
网帖 #2 及 #1 微博 #3 在 #1 网上 #1 热传 #4
一路 #2 风尘 #1 仆仆 #3 一路 #2 款款 #1 深情 #4
梁建红 #2 几赴 #1 京城 #2 替儿 #1 讨薪 #4
```

## Step 2: Training

``` sh
dir=exp
python train.py --gpu 0 --num_prosody 5 \
  --lr 0.001 \
  --num_epochs 10 \
  --batch_size 32 \
  --log_interval 10 \
  --train_data data/biaobei/train.txt \
  --cv_data data/biaobei/cv.txt \
  --model_dir $dir
```

## Step 3: Test

``` sh
dir=exp
python test.py --num_prosody 5 \
  --batch_size 32 \
  --test_data data/biaobei/test.txt \
  --checkpoint $dir/9.pt
```

The `test.py` will output the final test metric in the following style:

```
class   precision       recall      f1-score
#0      0.953965        0.962748    0.958336
#1      0.734226        0.799168    0.765321
#2      0.608731        0.478095    0.535561
#3      0.632626        0.646341    0.639410
#4      0.994012        0.996000    0.995004
```
