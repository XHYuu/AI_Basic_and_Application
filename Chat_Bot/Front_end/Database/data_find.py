import os
import json
import glob
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict


def create_data():
    dataset_dir = "dataset"
    json_files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))

    merged_data = []
    current_id = 1

    # 合并所有文件内容
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for item in items:
                item["id"] = f"{current_id:04d}"
                merged_data.append(item)
                current_id += 1

    # 保存为合并后的 data.json 文件
    output_path = "data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)


def create_database(model):
    create_data()
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # 定义类别集合
    labels = set(item["label"] for item in data)
    dim = 384  # 模型输出维度
    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)

    # 每个类别的向量索引 + 元数据
    label_to_index = {}
    label_to_data = defaultdict(list)

    # 建索引
    for label in labels:
        label_to_index[label] = faiss.IndexFlatL2(dim)

    for item in data:
        question = item["question"]
        label = item["label"]
        vec = model.encode([question], convert_to_numpy=True).astype("float32")

        label_to_index[label].add(vec)
        label_to_data[label].append(item)

    # 保存所有索引和元数据
    for label in labels:
        index_path = os.path.join(index_dir, f"{label}_index.bin")
        faiss.write_index(label_to_index[label], index_path)

        meta_path = os.path.join(index_dir, f"{label}_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(label_to_data[label], f)

    print("✅ 所有类别的索引与元数据构建完成！")


def search_by_label(query, label, root_path="faiss_index/", top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # create_database(model)
    index_path = root_path + f"{label}_index.bin"
    meta_path = root_path + f"{label}_meta.pkl"

    if not os.path.exists(index_path): print("Database error: With out find database!")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_vector, top_k * 2)  # 多取一些，便于后续去重

    seen_questions = set()
    result = []

    for idx in I[0]:
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx]
        if item["question"] in seen_questions:
            continue
        seen_questions.add(item["question"])
        result.append(item)
        if len(result) >= top_k:
            break

    return result


if __name__ == "__main__":
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    result = search_by_label("I am experiencing symptoms of mild fever, headache, and body aches. I recently took a trip to an area with high risk of Lyme disease. Should I start taking antibiotics?", "symptom-description", top_k=10)
    print(result[0])
