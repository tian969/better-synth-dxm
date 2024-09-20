# _*_ coding: utf-8 _*_
"""
Time: 2024/8/8 17:38
Author: Jingjun Zhang
Describe: 
"""
import json
import numpy as np
import argparse
from sklearn.cluster import KMeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, help="data/embedding/input_data.npy")
    parser.add_argument('--instruction_path', type=str, help="data/input_data.json")
    parser.add_argument('--save_path', type=str, help="output/output_kmeans_100.json")
    args = parser.parse_args()

    EMBEDDING_PATH = args.embedding_path
    INSTRUCTION_PATH = args.instruction_path
    SAVE_PATH = args.save_path

    embeddings = []
    embeddings.extend(np.load(f'{EMBEDDING_PATH}'))

    print(len(embeddings))
    print("K-MEANS")

    # KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)


    # kmeans.cluster_centers_
    def find_nearest(embedding, embeddings):
        distances = ((embeddings - embedding) ** 2).sum(axis=1)
        return distances.argmin()


    def distances_score(embedding, embeddings):
        distances = ((embeddings - embedding) ** 2).sum(axis=1)
        return distances


    cluster_center_indices = [find_nearest(center, embeddings) for center in kmeans.cluster_centers_]

    print(cluster_center_indices)

    data = []
    with open(f"{INSTRUCTION_PATH}", "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)

    # 构建簇字典
    cluster_dict = {}
    for i, label in enumerate(kmeans.labels_):
        label = int(label)
        if label not in cluster_dict:
            cluster_dict[label] = []

        # 计算数据点与簇中心的距离
        distance_to_center = round(distances_score(embeddings[i], kmeans.cluster_centers_[label].reshape(1, -1))[0], 2)

        # 将数据点和距离添加到簇字典中
        cluster_dict[label].append({
            "data": data[i],
            "distance_to_center": float(distance_to_center)
        })

    # 将簇字典转换为JSON格式并保存
    cluster_json = json.dumps(cluster_dict, indent=4)
    with open(f'{SAVE_PATH}', 'w') as f:
        f.write(cluster_json)
