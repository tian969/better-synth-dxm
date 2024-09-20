#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/4 10:08
# @Author  : tian
# @File    : filter.py
# @Desc    : 根据当前筛选出来的数据id 从internvl重新构造的数据中选择

import json

ids = []
with open("../data/dj_internvl_50.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        ids.append(data["id"])

with open("../data/pretrain_data_v2_1.jsonl", 'r', encoding="utf-8") as f_in , \
    open("../data/pretrain_data_v3_2.jsonl", 'w', encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line)
        if data["id"] in ids:
            f_out.write(json.dumps(data) + "\n")
