#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 14:05
# @Author  : tian
# @File    : preprocess_data.py
# @Desc    : 修改源数据格式以适应聚类输入

import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            text = data['text']
            cleaned_text = text.replace('<__dj__image>\n', '').replace('<|__dj__eoc|>', '').strip()
            new_data = {
                "question": data['images'][0],
                "answer": cleaned_text
            }
            outfile.write(json.dumps(new_data) + '\n')

if __name__ == "__main__":
    input_file = '../data/mgm_pretrain_stage_2.jsonl'  
    output_file = '../data/julei.jsonl' 
    process_jsonl(input_file, output_file)