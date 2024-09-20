#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/4 9:45
# @Author  : tian
# @File    : dynamic_increase.py
# @Desc    : 保证多样性: 对internvl重写的数据集, 根据聚类结果实现动态抽取

import json


def parse_jsonl_file(file_path):
    cluster_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            cluster_id = data['cluster_id']
            entry = {
                "id": data['id'],
                "distance_to_center": data['distance_to_center']
            }

            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []

            cluster_dict[cluster_id].append(entry)

    return cluster_dict


file_path = '../data/dj_internvl_50.jsonl'  # 替换为合成后的文件路径
result_dict = parse_jsonl_file(file_path)
# print(result_dict)

keys = [int(k) for k in result_dict.keys()]
keys.sort()
need = [0] * 50  # 0~49 每个簇需要多少

for key in keys:
    need[key] = len(result_dict[str(key)]) // 4 + 1

print(sum(need))  # 一共5035条数据

import math

to_fill = [[] for _ in range(50)]
for idx, n in enumerate(need):  # 为每个簇补数据
    print(n)
    need_max = math.ceil(n / 4)
    need_min = math.floor(n / 4)
    need_mid = n - need_max - need_min
    # print(need_max, need_min, need_mid)
    to_get = result_dict[str(idx)]
    to_get.sort(key=lambda x: x['distance_to_center'])
    # print(to_get)
    to_fill[idx].extend(to_get[:need_min])
    to_fill[idx].extend(to_get[-need_max:])
    to_random_select = to_get[need_min:-need_max]
    # 在不与之前重复的情况下从中间随机抽取need_mid个
    import random

    random.shuffle(to_random_select)
    to_fill[idx].extend(to_random_select[:need_mid])

ids = []
for i in range(50):
    for j in range(len(to_fill[i])):
        ids.append(to_fill[i][j]['id'])

# 挑选出ids 然后从原数据挑选
with open('../data/pretrain_data_v3_3.jsonl', 'w') as f_out, \
        open('../data/dj_internvl_50.jsonl', 'r') as f_in:
    for line in f_in:
        data = json.loads(line)
        if data['id'] in ids:
            f_out.write(json.dumps(data) + "\n")
