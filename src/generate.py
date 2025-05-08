# src/generate.py
import torch
from src.model import RNNModel
from src.data_loader import build_vocab
from config import DATA_FILE, EMBEDDING_DIM, HIDDEN_DIM, GENERATE_LENGTH


def generate_poem(model, char_to_idx, idx_to_char, start_str, length=GENERATE_LENGTH):
    model.eval()
    hidden = model.init_hidden(batch_size=1)
    input_seq = [char_to_idx[char] for char in start_str]
    input_seq = torch.tensor([input_seq], dtype=torch.long)

    generated = start_str
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        last_char_logits = output[0, -1, :]
        last_char_probs = torch.softmax(last_char_logits, dim=0)
        next_char_idx = torch.multinomial(last_char_probs, 1).item()
        next_char = idx_to_char[next_char_idx]
        generated += next_char
        input_seq = torch.tensor([[char_to_idx[next_char]]], dtype=torch.long)

    return generated


def load_model_and_vocab():
    char_to_idx, idx_to_char, vocab_size = build_vocab(DATA_FILE)
    model = RNNModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load("model.pth"))
    return model, char_to_idx, idx_to_char


if __name__ == "__main__":
    model, char_to_idx, idx_to_char = load_model_and_vocab()
    start_str = "欲植滅度根"
    generated_poem = generate_poem(model, char_to_idx, idx_to_char, start_str)
    print("Generated Poem:")
    print(generated_poem)
