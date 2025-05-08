# src/data_loader.py
import json
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def preprocess_data(data):
    # 将所有文本合并为一个长字符串
    text = ""
    for item in data:
        text += item["title"] + "\n"
        for paragraph in item["paragraphs"]:
            text += paragraph + "\n"
    return text


def build_vocab(file_path):
    # 构建字符到索引的映射
    data = load_data(file_path)
    text = preprocess_data(data)
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    return char_to_idx, idx_to_char, vocab_size


def encode_text(text, char_to_idx):
    # 将文本转换为索引序列
    encoded_text = [char_to_idx[char] for char in text]
    return encoded_text


class TextDataset(Dataset):
    def __init__(self, encoded_text, seq_length):
        self.encoded_text = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        seq = self.encoded_text[idx : idx + self.seq_length]
        target = self.encoded_text[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


def get_dataloader(file_path, seq_length, batch_size):
    data = load_data(file_path)
    text = preprocess_data(data)
    char_to_idx, idx_to_char, vocab_size = build_vocab(text)
    encoded_text = encode_text(text, char_to_idx)
    dataset = TextDataset(encoded_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, char_to_idx, idx_to_char, vocab_size
