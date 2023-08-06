import os
import sys
import random

# 指定这些文件名的数据，作为测试集；
test_sets = [
    "Abeiduo-0000_kw7rtfpo4dmxbdmgx4ryfezznc6oo4i_000001",
    "Babala-0000_hxj4a3e2aw2oundfm7ddctfs8nqzake_000077",
    "Bachongshenzi-0000_j8bsdnkvi7nnv5fygudeymouxoir04p_000141",
    "Bannite-0000_jna16rusizvz71pjygrsm2ihhzxmy7o_000282",
    "Chongyun-0000_sxtn0zmwf70u62sdkljoz4xovcrrzks_000370",
    "Dadalian-0000_fmrlyvhwpxu082czmruxa4kgdoew7c5_000428",
    "Diaona-0000_f032apxfqa231wdgb2ugupgntreo8zr_000487",
    "Diluke-0000_k33bresfktz04phbhtkallqz234oea3_000593",
    "Feixieer-0006_7xqxl6v6lxkdmtohhnqbi91wxxhncsy_000668",
    "Fengyuanwanye-0000_pfu7mnyysrat310mhrm6az5m6e9ymw4_000700",
    "Ganyu-0036_ps22nhruh62jyqg4fzn88xvsjegnm83_000885",
    "Huanglongyidou-0000_7jr19i05efuan5wy4p9sv9r3syr3gqh_000943",
    "Hutao-0000_73ne3fbhw4emno2z3ss6wf7zjdmpph9_001063",
    "Jiutiaoshaluo-0003_r18ywz4fgjxfhqs90yst7k5f7am4ta7_001158",
    "Keli-0000_icnus2c6r8piuvz8k9uz2i5tt0w9ydb_001281",
    "Keqing-0000_pwfdclndgm1ko3aldw7exakv3m1okz0_001373",
    "Leidianjiangjun-0005_gbk463w5dkebim9fe1a1szqrgryl93r_001473",
    "Lisha-0002_s9z1c15d09jktqmzx68sg2gki33bi20_001571",
    "Luoshaliya-0011_4c72ikrmyigd5t7uexxrd4fs236k7pl_001634",
    "Mona-0000_oigsdo9qjdx0y69kibx756x3z6fh106_001697",
    "Ningguang-0000_aop2mtdmoe6nv80quopzx6upqn192fl_001773",
    "Nuoaier-0000_h4idjj9ayd0qo12u6rghfjm7a75n71l_001852",
    "Qin-0000_2fb1097nmz3pylf9rni0fs6l6g92xnm_001967",
    "Qiqi-0000_o90h18wb0xuivuw6cw7wtmp8wotjzau_002034",
    "Shanhugongxinhai-0000_0k65v3e7bjw1ka7oqzhthk4lg3t6spn_002084",
    "Shatang-0041_5frhvfx5egkw3icidyomhhsprkhyn0q_002282",
    "Shenhe-0000_3d4fhyppxc49nztnowjmtjubvs58amm_002286",
    "Shenlilinghua-0000_hx23f7cf96qrxdedj1x8km0ksmc8fwq_002401",
    "Xiangling-0000_j816336nczz00zb3kqzxxnuve3ub5w2_002528",
    "Xiaogong-0000_jqcxe1bc0knjaxxoj5ztkyewbck49l7_002613",
    "Xinyan-0000_m9dmptn3n55dm3f2usz49fnno916knc_002741",
    "Yanfei-0000_1uib59lusm39lajs3huhxoyb4hc9l30_002854",
    "Yelan-0000_kthng1wgpyyj8he68tx8cp87791wv5l_002970",
    "Youla-0000_qdukmlynp0obxk2qs7h7yhzz5v86n6u_003085",
]

if len(sys.argv) != 7:
    print('Usage: prepare_data.py lexicon in_data_dir all_data_path '
          'test_data_path valid_data_path train_data_path')
    sys.exit(-1)

lexicon = {}
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.strip().split()
        lexicon[arr[0]] = arr[1:]

cnt = 0
test_find = []
with open(sys.argv[2], encoding='utf8') as fin, \
     open(sys.argv[3], 'w', encoding='utf8') as fout_all, \
     open(sys.argv[4], 'w', encoding='utf8') as fout_test, \
     open(sys.argv[5], 'w', encoding='utf8') as fout_valid, \
     open(sys.argv[6], 'w', encoding='utf8') as fout_train:

    lines = [x.strip() for x in fin.readlines()]
    random.shuffle(lines)

    for line in lines:

        speaker, duration, text, pinyin_list, wav_path = line.split(' ')
        uttid = os.path.basename(wav_path).replace(".wav", "")

        if speaker == "spk":  # 跳过首行
            continue

        # 跳过含英文的case
        skip_cases = ["UP", "B", "O", "live", "8"]
        skip = False
        for case in skip_cases:
            if case in text:
                skip = True
        if skip is True:
            continue

        phones = ["sil"]
        for x in pinyin_list.split(','):
            if x in ["#0", "#1", "#2", "#3"]:
                phones.append(x)
            elif x in lexicon:
                phones.extend(lexicon[x])
            elif x == "n2":
                phones.extend(lexicon["en2"])
            else:
                print('{} \n{} \nOOV {}'.format(text, pinyin_list, x))
                sys.exit(-1)
        phones.append("sil")

        write_line = '{}|{}|{}\n'.format(wav_path, speaker, ' '.join(phones))

        fout_all.write(write_line)

        if uttid in test_sets:
            fout_test.write(write_line)  # 指定测试集
            test_find.append(uttid)
        elif cnt < 100:
            fout_valid.write(write_line)  # 随机 100条作为验证集
            cnt += 1
        else:
            fout_train.write(write_line)  # 其余作为训练集
            cnt += 1

# 检查测试数据是否都已经放入到测试集中：
if len(test_sets) != len(test_find):
    print(f"test sets lens not equal: "
          f"len(test_sets) = {len(test_sets)} vs "
          f"len(test_find) = {len(test_find)}")

    for item in test_sets:
        if item not in test_find:
            print(f"Test uttid {item} is not finded.")
