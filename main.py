# main.py
import argparse
from src.train import train_model
from src.generate import generate_poem, load_model_and_vocab


def main():
    parser = argparse.ArgumentParser(description="Poem Generator")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--generate", action="store_true", help="Generate a poem")
    parser.add_argument(
        "--start",
        type=str,
        default="欲植滅度根",
        help="Starting string for poem generation",
    )
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.generate:
        model, char_to_idx, idx_to_char = load_model_and_vocab()
        generated_poem = generate_poem(model, char_to_idx, idx_to_char, args.start)
        print("Generated Poem:")
        print(generated_poem)
    else:
        print("Please specify --train or --generate")


if __name__ == "__main__":
    main()
