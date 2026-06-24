#!/usr/bin/env python
"""Prepare one-shot MultiClinSum clinical-report evaluation files."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from config import MODEL_PATH_MAP, RETRIEVER_MODEL_PATH_MAP  # noqa: E402


DEFAULT_DATASET_DIR = Path(
    "/home/wenting/workspace/dataset/multiclinsum/multiclinsum_test_en"
)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

PROMPT_TEMPLATE = """You are an expert medical professional.
####
Summarize the given clinical case report into a discharge summary of 100 words or less. Use the examples to guide word choice.

Clinical Case Report 1:
{demo_sd}

Discharge Summary 1:
{demo_rs}
##
Clinical Case Report 2:
{test_sd}

Discharge Summary 2:"""


def str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tokenizer", default="Qwen2.5-7B", choices=sorted(MODEL_PATH_MAP))
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--oov_prefix_space",
        type=str_to_bool,
        default=True,
        help="Prefix each unigram with a space before tokenizer fragmentation checks.",
    )
    parser.add_argument(
        "--retriever_backend",
        choices=["pubmedbert", "token_jaccard"],
        default="pubmedbert",
    )
    parser.add_argument("--retriever_model_path", default=str(RETRIEVER_MODEL_PATH_MAP["medical_pubmedbert"]))
    parser.add_argument("--retriever_batch_size", type=int, default=16)
    parser.add_argument("--retriever_max_length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "")


def default_output_dir(model_name: str) -> Path:
    return REPO_ROOT / f"outputs_{model_slug(model_name)}" / "inference_data" / "medical"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def numeric_id(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else -1


def load_multiclinsum_pairs(dataset_dir: Path) -> list[dict[str, object]]:
    fulltext_dir = dataset_dir / "fulltext"
    summaries_dir = dataset_dir / "summaries"
    if not fulltext_dir.exists():
        raise FileNotFoundError(f"Missing fulltext directory: {fulltext_dir}")
    if not summaries_dir.exists():
        raise FileNotFoundError(f"Missing summaries directory: {summaries_dir}")

    records = []
    for sd_path in sorted(fulltext_dir.glob("*.txt"), key=numeric_id):
        rs_path = summaries_dir / f"{sd_path.stem}_sum.txt"
        if not rs_path.exists():
            raise FileNotFoundError(f"Missing summary for {sd_path.name}: {rs_path}")
        records.append(
            {
                "id": sd_path.stem,
                "numeric_id": numeric_id(sd_path),
                "SD": read_text(sd_path),
                "RS": read_text(rs_path),
            }
        )
    if not records:
        raise ValueError(f"No paired MultiClinSum records found under: {dataset_dir}")
    return records


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def oov_concentration(
    text: str,
    tokenizer,
    fragment_cache: dict[str, bool],
    prefix_space: bool,
) -> float:
    tokens = words(text)
    if not tokens:
        return 0.0

    fragmented = 0
    for token in tokens:
        cache_key = f"{int(prefix_space)}\t{token}"
        is_fragmented = fragment_cache.get(cache_key)
        if is_fragmented is None:
            probe = f" {token}" if prefix_space else token
            token_ids = tokenizer.encode(probe, add_special_tokens=False)
            is_fragmented = len(token_ids) > 1
            fragment_cache[cache_key] = is_fragmented
        if is_fragmented:
            fragmented += 1
    return fragmented / len(tokens)


def attach_oov_scores(records: list[dict[str, object]], tokenizer, prefix_space: bool) -> None:
    cache: dict[str, bool] = {}
    for record in records:
        record["oov_sd"] = oov_concentration(str(record["SD"]), tokenizer, cache, prefix_space)
        record["oov_rs"] = oov_concentration(str(record["RS"]), tokenizer, cache, prefix_space)


def split_high_oov(
    records: list[dict[str, object]],
    score_key: str,
    test_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked = sorted(records, key=lambda row: (-float(row[score_key]), int(row["numeric_id"])))
    test_ids = {row["id"] for row in ranked[:test_count]}
    test_records = [row for row in ranked[:test_count]]
    train_records = [row for row in records if row["id"] not in test_ids]
    return train_records, test_records


def split_random(
    records: list[dict[str, object]],
    test_count: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    test_ids = {row["id"] for row in shuffled[:test_count]}
    test_records = sorted(shuffled[:test_count], key=lambda row: int(row["numeric_id"]))
    train_records = [row for row in records if row["id"] not in test_ids]
    return train_records, test_records


def resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts_pubmedbert(
    texts: list[str],
    model_path: str,
    batch_size: int,
    max_length: int,
    device_name: str,
):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        embeddings.append(pooled.cpu())
        print(f"embedded {min(start + batch_size, len(texts))}/{len(texts)} texts")

    return torch.cat(embeddings, dim=0)


def word_set(text: str) -> set[str]:
    return {token.lower() for token in words(text)}


def best_demo_indices_token_jaccard(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
) -> list[int]:
    train_sets = [word_set(str(row["SD"])) for row in train_records]
    indices = []
    for test in test_records:
        test_set = word_set(str(test["SD"]))
        best_idx = 0
        best_score = -1.0
        for idx, train_set in enumerate(train_sets):
            union_size = len(test_set | train_set)
            score = 0.0 if union_size == 0 else len(test_set & train_set) / union_size
            if score > best_score:
                best_idx = idx
                best_score = score
        indices.append(best_idx)
    return indices


def best_demo_indices_pubmedbert(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    embedding_by_id: dict[str, object],
) -> list[int]:
    import torch

    train_embeddings = torch.stack([embedding_by_id[str(row["id"])] for row in train_records])
    test_embeddings = torch.stack([embedding_by_id[str(row["id"])] for row in test_records])
    similarities = torch.matmul(test_embeddings, train_embeddings.T)
    return similarities.argmax(dim=1).tolist()


def best_demo_indices(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    args: argparse.Namespace,
    embedding_by_id: dict[str, object] | None,
) -> list[int]:
    if args.retriever_backend == "token_jaccard":
        return best_demo_indices_token_jaccard(train_records, test_records)
    if embedding_by_id is None:
        raise ValueError("PubMedBERT retrieval requires precomputed embeddings")
    return best_demo_indices_pubmedbert(train_records, test_records, embedding_by_id)


def make_prompt(demo: dict[str, object], test: dict[str, object]) -> str:
    return PROMPT_TEMPLATE.format(
        demo_sd=str(demo["SD"]).strip(),
        demo_rs=str(demo["RS"]).strip(),
        test_sd=str(test["SD"]).strip(),
    )


def build_icl_rows(
    train_records: list[dict[str, object]],
    test_records: list[dict[str, object]],
    args: argparse.Namespace,
    embedding_by_id: dict[str, object] | None,
) -> list[dict[str, str]]:
    demo_indices = best_demo_indices(train_records, test_records, args, embedding_by_id)
    rows = []
    for test, demo_idx in zip(test_records, demo_indices):
        demo = train_records[demo_idx]
        rows.append(
            {
                "icl_input": make_prompt(demo, test),
                "target": str(test["RS"]).strip(),
            }
        )
    return rows


def write_json(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(args.tokenizer)
    )
    tokenizer_path = (
        Path(args.tokenizer_path).expanduser().resolve()
        if args.tokenizer_path
        else Path(MODEL_PATH_MAP[args.tokenizer]).expanduser().resolve()
    )

    if not 0 < args.test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {args.test_fraction}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    print(f"dataset_dir: {dataset_dir}")
    print(f"output_dir: {output_dir}")
    print(f"oov_tokenizer: {tokenizer_path}")
    print(f"retriever_backend: {args.retriever_backend}")

    records = load_multiclinsum_pairs(dataset_dir)
    test_count = max(1, math.ceil(len(records) * args.test_fraction))
    print(f"records: {len(records)}")
    print(f"test_count: {test_count}")

    oov_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    attach_oov_scores(records, oov_tokenizer, args.oov_prefix_space)

    embedding_by_id = None
    if args.retriever_backend == "pubmedbert":
        print(f"embedding all records for retrieval: {len(records)}")
        embeddings = embed_texts_pubmedbert(
            [str(row["SD"]) for row in records],
            args.retriever_model_path,
            args.retriever_batch_size,
            args.retriever_max_length,
            args.device,
        )
        embedding_by_id = {
            str(record["id"]): embeddings[index] for index, record in enumerate(records)
        }

    scenarios = {
        "highOOV_SD": (
            "ClinReport_1_highOOV_SD.json",
            *split_high_oov(records, "oov_sd", test_count),
        ),
        "highOOV_RS": (
            "ClinReport_1_highOOV_RS.json",
            *split_high_oov(records, "oov_rs", test_count),
        ),
        "random": (
            "ClinReport_1_random.json",
            *split_random(records, test_count, args.seed),
        ),
    }

    for scenario, (filename, train_records, test_records) in scenarios.items():
        print(f"\npreparing {scenario}: train={len(train_records)} test={len(test_records)}")
        rows = build_icl_rows(train_records, test_records, args, embedding_by_id)
        output_path = output_dir / filename
        write_json(output_path, rows)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
