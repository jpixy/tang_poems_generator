# src/train.py
import torch
import torch as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import RNNModel
from src.data_loader import get_dataloader
from config import (
    DATA_FILE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    SEQ_LENGTH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
import torch.nn as nn  # 确保正确导入


def train_model():
    dataloader, char_to_idx, idx_to_char, vocab_size = get_dataloader(
        DATA_FILE, SEQ_LENGTH, BATCH_SIZE
    )
    model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()  # 使用正确的导入
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (seq, target) in enumerate(dataloader):
            current_batch_size = seq.size(0)  # 获取当前批次的实际大小
            hidden = model.init_hidden(
                batch_size=current_batch_size
            )  # 动态调整 hidden 的大小

            optimizer.zero_grad()
            output, hidden = model(seq, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            loss = criterion(output.transpose(1, 2), target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    torch.save(model.state_dict(), "model.pth")
    return model, char_to_idx, idx_to_char


if __name__ == "__main__":
    train_model()

