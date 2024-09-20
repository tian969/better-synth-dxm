#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

def merge_jsonl(file1, file2, output_file):
    # 读取第一个文件并构建字典
    data1 = {}
    with open(file1, 'r') as f1:
        for line in f1:
            record = json.loads(line)
            data1[record['image']] = record

    # 读取第二个文件并合并数据
    with open(file2, 'r') as f2, open(output_file, 'w') as outfile:
        for line in f2:
            record2 = json.loads(line)
            image = record2['image']
            if image in data1:
                record1 = data1[image]
                merged_record = {**record1, **record2}
                outfile.write(json.dumps(merged_record) + '\n')
            else:
                print(f"Warning: Image {image} not found in {file1}")

# 使用示例
file1 = 'data/fg_anno.jsonl'
file2 = 'data/hal_from_desc.jsonl'
output_file = 'data/your.jsonl'
merge_jsonl(file1, file2, output_file)
