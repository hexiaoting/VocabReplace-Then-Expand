#!/usr/bin/env python
"""Select top medical-domain tokens for a base tokenizer."""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
S1_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "data"
    / "pretraining-data"
    / "100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt"
)
DEFAULT_BPE_DIR = S1_DIR / "VocabFiles-medical" / "bpe_256000"
DEFAULT_TOP_K = 10_000
TOKEN_PATTERN = re.compile(r"^[A-Za-zĠ]+$")

sys.path.insert(0, str(REPO_ROOT))
from config import MODEL_PATH_MAP  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, choices=sorted(MODEL_PATH_MAP))
    parser.add_argument("--input_path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--bpe_dir", default=str(DEFAULT_BPE_DIR))
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "")


def token_text(token: str) -> str:
    return token.replace("Ġ", " ")


def is_english_token(token: str) -> bool:
    return bool(TOKEN_PATTERN.fullmatch(token)) and bool(token.replace("Ġ", ""))


def count_domain_tokens(input_path: Path, vocab_path: Path, merges_path: Path) -> Counter:
    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    counts = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            counts.update(tokenizer.encode(line).ids)

    return counts


def select_tokens(counts: Counter, vocab: dict[str, int], base_tokenizer, top_k: int):
    id_to_token = {idx: token for token, idx in vocab.items()}
    base_vocab = base_tokenizer.get_vocab()
    selected = []

    for token_id, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if token_id < 256:
            continue

        token = id_to_token.get(token_id)
        if token is None or token in base_vocab:
            continue
        if not is_english_token(token):
            continue
        if len(base_tokenizer.tokenize(token_text(token))) <= 1:
            continue

        selected.append((token, token_id, freq))
        if len(selected) == top_k:
            break

    return selected


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    bpe_dir = Path(args.bpe_dir).expanduser().resolve()
    vocab_path = bpe_dir / "vocab.json"
    merges_path = bpe_dir / "merges.txt"
    model_path = Path(MODEL_PATH_MAP[args.tokenizer]).expanduser().resolve()

    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")
    if not input_path.exists():
        raise FileNotFoundError(f"Training corpus does not exist: {input_path}")
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Missing BPE files under: {bpe_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {model_path}")

    print("\n------------------------------")
    print(f"Tokenizer : {args.tokenizer}")
    print(f"Model path: {model_path}")
    print(f"BPE dir   : {bpe_dir}")
    print(f"Input path: {input_path}")
    print(f"Top K     : {args.top_k}")

    domain_tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    base_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    selected = select_tokens(
        count_domain_tokens(input_path, vocab_path, merges_path),
        domain_tokenizer.get_vocab(),
        base_tokenizer,
        args.top_k,
    )

    output_dir = S1_DIR / f"../outputs_{model_slug(args.tokenizer)}" / "selected_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"top_{args.top_k}_tokens_with_freq.tsv"

    with output_path.open("w", encoding="utf-8") as f:
        f.write("token\ttoken_id\tfrequency\n")
        for token, token_id, freq in selected:
            f.write(f"{token}\t{token_id}\t{freq}\n")

    print(f"Selected  : {len(selected)}")
    print(f"Saved     : {output_path}")


if __name__ == "__main__":
    main()
