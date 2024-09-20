#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/5
# @Author  : tian
# @File    : batch_infer_intern_vl2
# @Desc    : 数据合成相关 使用LMDeploy框架进行批量推理

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import os
import time
import glob
from tqdm import tqdm
import json
import math

# 替换为具体的模型路径
model = '/home/disk2/zhangjingjun/data_syn/caches/models/InternVL2-40B'
chat_template_config = ChatTemplateConfig('internvl-zh-hermes2')
chat_template_config.meta_instruction = system_prompt
# config = TurbomindEngineConfig()
pipe = pipeline(model, chat_template_config=chat_template_config,
                backend_config=TurbomindEngineConfig(session_len=4096, tp=4))

# 替换为具体的图像路径
image_folder = "/home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/images"

# 指定读入文件路径
file_1 = "/home/disk2/zhangjingjun/data_syn/dj_synth_challenge/input/pretrain_stage_1/0902/recaption/mgm_pretrain_0903_gx.jsonl"
output_file = "mgm_pretrain_0903_internvl.jsonl"
json_objs = []

with open(file=file_1, mode='r', encoding='utf-8') as file:
    for line in file:
        try:
            json_obj = json.loads(line)
            json_objs.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"解析JSON时出错: {e}")
            continue
img_urls = [os.path.join(image_folder, js['image']) for js in json_objs]

# 用intern-VL2-40B对图片进行详细描述的prompt
# describe_prompt = "You are a powerful image captioner. Please describe this picture. Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the ontents by itemizing them in list form. Minimize aesthetic descriptionsas much as possible. You can make reasonable inferences about the relevant information in the image."

# 用intern-VL2-40B对图片进行简洁描述的prompt
describe_prompt = """
Expert in image description, please provide a concise and precise summary of the main subject in the image as per the following guidelines:

Avoid starting with phrases like "The image..."; directly describe the main object.

Keep the description extremely brief! Focus only on the primary content, without delving into details.

If there is text, prioritize describing the main visible text at a glance, without elaborating on other text.

"""
prompts = []
index = 0
try:
    for img_url in img_urls:
        index += 1
        print(index, '\t', img_url)
        prompts.append((describe_prompt, load_image(img_url), img_url))
except Exception as e:
    print(e)

# prompts = [(describe_prompt, load_image(img_url), img_url) for img_url in img_urls]
batch_size = 100


def get_batches(data, bsz):
    n = len(data)
    for i in range(0, n, bsz):
        yield data[i:i + bsz]


batches = get_batches(prompts, batch_size)
# 替换输出路径
with open(output_file, "w", encoding="utf-8") as f_out:
    for batch in tqdm(batches, total=math.ceil(len(prompts) / batch_size), desc="Processing"):
        urls = [image_url for _, _, image_url in batch]
        p = [(pr, i) for pr, i, _ in batch]
        response = pipe(p)
        for res, img_path in zip(response, urls):
            data = {
                "image": img_path,
                "description": res.text
            }
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
