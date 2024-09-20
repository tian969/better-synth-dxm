#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 19:36
# @Author  : tian
# @File    : insert_cluster.py
# @Desc    : 将聚类结果写入新的jsonl文件
import json

# 读取JSON文件并转换为字典
with open('../data/output_50_dict.json', 'r') as f:
    image_dict = json.load(f)

# 读取jsonl文件并处理每一行
with open('../data/mgm_pretrain_stage_2.jsonl', 'r', encoding="utf-8") as f:
    lines = f.readlines()

f_out = open('../data/output_50_with_cluster.jsonl', 'w')
# 处理每一行
for line in lines:
    data = json.loads(line)
    image_path = data['images'][0]
    if image_path in image_dict:
        cluster_info = image_dict[image_path]
        _map = {}
        _map['cluster'] = cluster_info['cluster']
        _map['distance_to_center'] = cluster_info['distance_to_center']
        data['cluster'] = _map
        # 输出处理后的行
        f_out.write(json.dumps(data) + "\n")