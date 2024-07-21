import os
import tiktoken
import numpy as np
import random

def read_file(file_path):
    """读取文件并返回非空行的列表"""
    entries = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip() and len(line) > 2:
                    entries.append(line)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")
    return entries

def shuffle_and_split(entries, split_ratio=0.9):
    """打乱并拆分数据"""
    random.shuffle(entries)
    n = len(entries)
    train_entries = entries[: int(n * split_ratio)]
    val_entries = entries[int(n * split_ratio):]
    return train_entries, val_entries

def encode_data(entries, encoder):
    """编码数据"""
    data = " ".join("{}".format(entry) for entry in entries)
    return encoder.encode_ordinary(data)

def save_to_file(data, file_path):
    """保存数据到文件"""
    try:
        data.tofile(file_path)
    except Exception as e:
        print(f"保存到 {file_path} 时发生错误: {e}")

def main():
    input_file_path = os.path.join(os.path.dirname(__file__), "data", "book.txt")
    entries = read_file(input_file_path)

    if not entries:
        return

    print(f"行数: {len(entries)}")

    train_entries, val_entries = shuffle_and_split(entries)

    enc = tiktoken.get_encoding("gpt2")
    train_ids = encode_data(train_entries, enc)
    val_ids = encode_data(val_entries, enc)

    print(f"训练集有 {len(train_ids):,} 个tokens")
    print(f"验证集有 {len(val_ids):,} 个tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    save_to_file(train_ids, os.path.join(os.path.dirname(__file__), "data", "train.bin"))
    save_to_file(val_ids, os.path.join(os.path.dirname(__file__), "data", "val.bin"))

if __name__ == "__main__":
    main()