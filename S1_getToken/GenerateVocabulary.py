#!/usr/bin/env python
"""Train the medical-domain ByteLevel BPE tokenizer."""

import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (REPO_ROOT / "data/pretraining-data/100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt")
DEFAULT_VOCAB_SIZE = 256_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--domain", default="medical")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(__file__).resolve().parent / f"VocabFiles-{args.domain}" / f"bpe_{args.vocab_size}"

    if not input_path.exists():
        raise FileNotFoundError(f"Training corpus does not exist: {input_path}")
    if args.vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {args.vocab_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n------------------------------")
    print(f"Input path : {input_path}")
    print(f"Vocab size : {args.vocab_size}")
    print(f"Output dir   : {output_dir}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(str(input_path), vocab_size=args.vocab_size, show_progress=True)
    tokenizer.save_model(str(output_dir))

    print("Done.")


if __name__ == "__main__":
    main()
