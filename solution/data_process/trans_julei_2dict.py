#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 16:52
# @Author  : tian
# @File    : trans_julei_2dict.py
# @Desc    : 聚类信息打印为dict


import json

def transform_json_to_dict(json_data):
    result_dict = {}
    for cluster_id, items in json_data.items():
        for item in items:
            question = item['data']['question']
            distance_to_center = item['distance_to_center']
            result_dict[question] = {
                "cluster": cluster_id,
                "distance_to_center": distance_to_center
            }
    return result_dict

json_data = json.load(open("../data/output_50.json", "r"))
# 转换JSON数据
result_dict = transform_json_to_dict(json_data)
with open("../data/output_50_dict.json", "w") as f:
    json.dump(result_dict, f, indent=4)