#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/9/3 15:53
# @Author  : tian
# @File    : ops.py
# @Desc    : 根据id做集合运算

import json
import argparse


def read_ids_from_jsonl(file_path):
    ids = set()
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            ids.add(data['id'])
    return ids


def save_ids_to_jsonl(ids, output_file):
    with open(output_file, 'w', encoding="utf-8") as file:
        for id in ids:
            file.write(json.dumps({'id': id}) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Perform set operations on JSONL files based on id attribute.')
    parser.add_argument('file1', type=str, help='Path to the first JSONL file')
    parser.add_argument('file2', type=str, help='Path to the second JSONL file')
    parser.add_argument('operation', type=str, choices=['difference', 'intersection', 'union'],
                        help='Set operation to perform')
    parser.add_argument('output', type=str, default="./output.jsonl", help='Path to the output JSONL file')

    args = parser.parse_args()

    ids1 = read_ids_from_jsonl(args.file1)
    ids2 = read_ids_from_jsonl(args.file2)

    if args.operation == 'difference':
        result = ids1 - ids2
    elif args.operation == 'intersection':
        result = ids1 & ids2
    elif args.operation == 'union':
        result = ids1 | ids2

    save_ids_to_jsonl(result, args.output)
    print(f"Result saved to {args.output}")


if __name__ == '__main__':
    main()
