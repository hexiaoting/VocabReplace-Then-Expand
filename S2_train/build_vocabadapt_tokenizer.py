#!/usr/bin/env python
"""Build a VOCABADAPT tokenizer from selected domain tokens."""

import argparse
import json
import pickle
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from config import MODEL_PATH_MAP  # noqa: E402

REPLACEMENT_ID_FILES = {
    "Llama3.1-8B": REPO_ROOT
    / "src"
    / "VocabularyConstruction"
    / "VOCABADAPT"
    / "TokenIDs_Can_Replace_Llama3.1-Magikarp+Unreachable.txt",
    "Qwen2.5-7B": REPO_ROOT
    / "src"
    / "VocabularyConstruction"
    / "VOCABADAPT"
    / "TokenIDs_Can_Replace_Qwen2.5-Magikarp+Unreachable.txt",
}


def str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "")


def default_selected_tokens_path(model_name: str) -> Path:
    return (
        REPO_ROOT
        / f"outputs_{model_slug(model_name)}"
        / "selected_tokens"
        / "top_10000_tokens_with_freq.tsv"
    )


def default_output_dir(model_name: str) -> Path:
    return REPO_ROOT / f"outputs_{model_slug(model_name)}" / "vocabadapt" / "tokenizer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, choices=sorted(MODEL_PATH_MAP))
    parser.add_argument("--selected_tokens_path", default=None)
    parser.add_argument("--top_k", type=int, default=10_000)
    parser.add_argument("--replace_unreachable_tokens", type=str_to_bool, default=True)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def read_selected_tokens(path: Path, top_k: int) -> list[str]:
    tokens = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            token = line.rstrip("\n").split("\t", 1)[0]
            if line_no == 0 and token == "token":
                continue
            if not token:
                continue
            tokens.append(token)
            if len(tokens) == top_k:
                break
    return tokens


def read_replacement_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def token_text(token: str) -> str:
    return " " + token[1:] if token.startswith("Ġ") else token


def collapse_with_known_merges(parts: list[str], known_merges: OrderedDict[str, list[str]]) -> list[str]:
    longest = ""
    start = -1
    end = -1
    for i in range(len(parts)):
        for j in range(i + 1, len(parts) + 1):
            candidate = "".join(parts[i:j])
            if candidate in known_merges and len(candidate) > len(longest):
                longest = candidate
                start = i
                end = j
    if not longest:
        return parts
    return parts[:start] + [longest] + parts[end:]


def build_merge_rules(tokens: list[str], base_tokenizer, base_model_vocab: dict[str, int]):
    merge_rules: OrderedDict[str, list[str]] = OrderedDict()
    for token in sorted(tokens, key=len):
        if token in base_model_vocab:
            continue
        parts = base_tokenizer.tokenize(token_text(token))
        if len(parts) <= 1:
            continue
        if len(parts) == 2:
            if token not in base_model_vocab and token not in merge_rules:
                merge_rules[token] = [parts[0], parts[1]]
            continue

        parts = collapse_with_known_merges(parts, merge_rules)
        if len(parts) == 2:
            if token not in base_model_vocab and token not in merge_rules:
                merge_rules[token] = [parts[0], parts[1]]
            continue

        merged = parts[0]
        for part in parts[1:]:
            left = merged
            right = part
            merged = f"{merged}{part}"
            if merged not in base_model_vocab and merged not in merge_rules:
                merge_rules[merged] = [left, right]
    return merge_rules


def shift_ids(obj, start_id: int, offset: int):
    if offset == 0:
        return obj
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == "id" and isinstance(value, int) and value >= start_id:
                obj[key] = value + offset
            elif key == "ids" and isinstance(value, list):
                obj[key] = [item + offset if isinstance(item, int) and item >= start_id else item for item in value]
            else:
                shift_ids(value, start_id, offset)
    elif isinstance(obj, list):
        for item in obj:
            shift_ids(item, start_id, offset)
    return obj


def shift_tokenizer_json(tokenizer_json: dict, start_id: int, offset: int) -> None:
    if offset == 0:
        return
    for added_token in tokenizer_json.get("added_tokens", []):
        token_id = added_token.get("id")
        if isinstance(token_id, int) and token_id >= start_id:
            added_token["id"] = token_id + offset
    for key, value in tokenizer_json.items():
        if key in {"model", "added_tokens"}:
            continue
        shift_ids(value, start_id, offset)


def shift_tokenizer_config(path: Path, start_id: int, offset: int) -> None:
    if offset == 0 or not path.exists():
        return
    data = json.load(path.open("r", encoding="utf-8"))
    decoder = data.get("added_tokens_decoder")
    if isinstance(decoder, dict):
        shifted = {}
        for key, value in decoder.items():
            try:
                token_id = int(key)
            except ValueError:
                shifted[key] = value
                continue
            if token_id >= start_id:
                token_id += offset
            shifted[str(token_id)] = value
        data["added_tokens_decoder"] = shifted
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def merge_pair(merge) -> list[str]:
    return merge.split() if isinstance(merge, str) else list(merge)


def merge_product(merge) -> str:
    return "".join(merge_pair(merge))


def serialize_merge(merge: list[str], as_string: bool):
    return " ".join(merge) if as_string else merge

def source_ids_for_token(base_tokenizer, token: str) -> list[int]:
    source_ids = base_tokenizer.encode(token_text(token), add_special_tokens=False)
    if source_ids:
        return source_ids
    return base_tokenizer.convert_tokens_to_ids(base_tokenizer.tokenize(token_text(token)))


def write_tsv(path: Path, rows: list[list[object]], header: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(item) for item in row) + "\n")


def main() -> None:
    args = parse_args()
    model_path = Path(MODEL_PATH_MAP[args.tokenizer]).expanduser().resolve()
    selected_tokens_path = (
        Path(args.selected_tokens_path).expanduser().resolve()
        if args.selected_tokens_path
        else default_selected_tokens_path(args.tokenizer)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(args.tokenizer)
    )

    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model/tokenizer path does not exist: {model_path}")
    if not selected_tokens_path.exists():
        raise FileNotFoundError(f"Selected tokens file does not exist: {selected_tokens_path}")
    if args.replace_unreachable_tokens and args.tokenizer not in REPLACEMENT_ID_FILES:
        raise ValueError(f"No replacement id file configured for {args.tokenizer}")

    selected_tokens = read_selected_tokens(selected_tokens_path, args.top_k)
    base_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer_json_path = model_path / "tokenizer.json"
    tokenizer_json = json.load(tokenizer_json_path.open("r", encoding="utf-8"))
    base_model_vocab = tokenizer_json["model"]["vocab"]
    base_merges = tokenizer_json["model"].get("merges", [])
    merges_as_string = bool(base_merges) and isinstance(base_merges[0], str)
    base_model_vocab_size = len(base_model_vocab)
    base_tokenizer_len = len(base_tokenizer)

    merge_rules = build_merge_rules(selected_tokens, base_tokenizer, base_model_vocab)
    merge_tokens = list(merge_rules.keys())

    replacement_id_file = REPLACEMENT_ID_FILES.get(args.tokenizer)
    replacement_ids = read_replacement_ids(replacement_id_file) if args.replace_unreachable_tokens else []
    replacement_count = min(len(replacement_ids), len(merge_tokens))
    replacement_ids = replacement_ids[:replacement_count]
    replacement_tokens = merge_tokens[:replacement_count]
    expansion_tokens = merge_tokens[replacement_count:]

    id_to_base_token = {idx: token for token, idx in base_model_vocab.items()}
    new_vocab = dict(base_model_vocab)
    replaced_rows = []
    replaced_pickle = []

    for token_id, new_token in zip(replacement_ids, replacement_tokens):
        old_token = id_to_base_token[token_id]
        if old_token in new_vocab:
            del new_vocab[old_token]
        new_vocab[new_token] = token_id
        source_ids = source_ids_for_token(base_tokenizer, new_token)
        replaced_rows.append([token_id, old_token, new_token, " ".join(str(i) for i in source_ids)])
        replaced_pickle.append([token_id, old_token, new_token, source_ids])

    next_id = base_model_vocab_size
    for token in expansion_tokens:
        if token in new_vocab:
            continue
        new_vocab[token] = next_id
        next_id += 1

    expansion_count = next_id - base_model_vocab_size
    replaced_old_tokens = {row[1] for row in replaced_rows}
    new_merges = [
        serialize_merge(merge, merges_as_string)
        for token, merge in merge_rules.items()
        if token in replacement_tokens
    ]
    filtered_base_merges = [merge for merge in base_merges if merge_product(merge) not in replaced_old_tokens]
    expansion_merges = [
        serialize_merge(merge, merges_as_string)
        for token, merge in merge_rules.items()
        if token in expansion_tokens
    ]
    tokenizer_json["model"]["vocab"] = new_vocab
    tokenizer_json["model"]["merges"] = new_merges + filtered_base_merges + expansion_merges
    shift_tokenizer_json(tokenizer_json, base_model_vocab_size, expansion_count)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_tokenizer.save_pretrained(str(output_dir))

    with (output_dir / "tokenizer.json").open("w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    if (output_dir / "vocab.json").exists():
        with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(new_vocab, f, ensure_ascii=False)
    if (output_dir / "merges.txt").exists():
        with (output_dir / "merges.txt").open("w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for merge in tokenizer_json["model"]["merges"]:
                f.write((merge if isinstance(merge, str) else " ".join(merge)) + "\n")

    shift_tokenizer_config(output_dir / "tokenizer_config.json", base_model_vocab_size, expansion_count)

    with (output_dir / "replaced_tokens.pkl").open("wb") as f:
        pickle.dump(replaced_pickle, f)
    write_tsv(output_dir / "replaced_tokens.tsv", replaced_rows, ["token_id", "old_token", "new_token", "source_ids"])

    final_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    final_tokenizer.save_pretrained(str(output_dir))

    build_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "base_model_path": str(model_path),
        "selected_tokens_path": str(selected_tokens_path),
        "top_k": args.top_k,
        "replace_unreachable_tokens": args.replace_unreachable_tokens,
        "replacement_id_file": str(replacement_id_file) if args.replace_unreachable_tokens else None,
        "output_dir": str(output_dir),
    }
    summary = {
        **build_config,
        "selected_token_count": len(selected_tokens),
        "merge_token_count": len(merge_tokens),
        "replacement_id_count": len(read_replacement_ids(replacement_id_file)) if args.replace_unreachable_tokens else 0,
        "replacement_count": replacement_count,
        "expansion_count": expansion_count,
        "base_model_vocab_size": base_model_vocab_size,
        "base_tokenizer_len": base_tokenizer_len,
        "final_model_vocab_size": len(new_vocab),
        "final_tokenizer_len": len(final_tokenizer),
    }

    with (output_dir / "build_config.json").open("w", encoding="utf-8") as f:
        json.dump(build_config, f, ensure_ascii=False, indent=2)
    with (output_dir / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n------------------------------")
    print(f"Tokenizer              : {args.tokenizer}")
    print(f"Selected tokens        : {len(selected_tokens)}")
    print(f"Merge tokens generated : {len(merge_tokens)}")
    print(f"Replacement enabled    : {args.replace_unreachable_tokens}")
    print(f"Replacement count      : {replacement_count}")
    print(f"Expansion count        : {expansion_count}")
    print(f"Final tokenizer len    : {len(final_tokenizer)}")
    print(f"Saved                  : {output_dir}")


if __name__ == "__main__":
    main()
