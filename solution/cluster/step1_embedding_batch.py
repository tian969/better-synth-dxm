# _*_ coding: utf-8 _*_
"""
Time: 2024/8/8 17:32
Author: Jingjun Zhang
Describe: 
"""
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModel
import json
from tqdm import tqdm
import numpy as np
import os
import argparse


def embed_texts_batched(texts, batch_size=30):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        tokens = {k: v.cuda() for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        all_embeddings.extend(embeddings)
    return all_embeddings


def load_sample(file_name):
    data = []
    with open(file_name, "r", encoding='utf-8') as file:
        for line in file:
            # 去除行末的换行符并解析JSON
            json_obj = json.loads(line.strip())
            data.append(json_obj)
        print(f"Data loaded: {file_name}.")

    ex_list = [[e["question"], e["answer"]] for e in data]
    ex_prompted = []
    for instruction, output in ex_list:
        try:
            # line = "Question:" + instruction + " Answer:" + output
            line = output
            ex_prompted.append(str(line))  # TODO 注意，一定要str类型
        except:
            continue
    return ex_prompted


# 初始化模型和分词器
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help="/home/disk2/zhangjingjun/data_syn/caches/models/bge")
    parser.add_argument('--instruction_path', type=str, help="data/input_data.json")
    parser.add_argument('--save_embedding_path', type=str, help="data/embedding/")
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    INSTRUCTION_PATH = args.instruction_path
    SAVE_EMBEDDING_PATH = args.save_embedding_path

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        # load_in_8bit=in_8bit,
    )
    model.eval()

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # load sample
    sample = load_sample(INSTRUCTION_PATH)
    print("START EMBEDDING ..." * 3)
    embeddings = embed_texts_batched(sample)
    print(len(embeddings))
    np.save(f'{SAVE_EMBEDDING_PATH}/{len(sample)}.npy', embeddings)