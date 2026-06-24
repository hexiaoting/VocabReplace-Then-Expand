"""Shared training utilities for CPTOnly and VOCABADAPT."""

from __future__ import annotations

import math
import re
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import datasets
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling

CHUNKING_VERSION = "simple_logical_chunking_v2"


def round_to_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def token_text(token: str) -> str:
    return " " + token[1:] if token.startswith("Ġ") else token


def source_ids_for_target_token(source_tokenizer, token: str) -> list[int]:
    source_vocab = source_tokenizer.get_vocab()
    if token in source_vocab:
        return [source_vocab[token]]
    source_ids = source_tokenizer.encode(token_text(token), add_special_tokens=False)
    if source_ids:
        return source_ids
    tokenized = source_tokenizer.tokenize(token_text(token))
    ids = source_tokenizer.convert_tokens_to_ids(tokenized)
    if isinstance(ids, int):
        return [ids]
    return [idx for idx in ids if idx is not None]


def initialize_embeddings_by_mean(model, source_tokenizer, target_tokenizer):
    source_input = model.get_input_embeddings().weight.detach().clone()
    source_output_layer = model.get_output_embeddings()
    source_output = source_output_layer.weight.detach().clone() if source_output_layer is not None else None

    model.resize_token_embeddings(len(target_tokenizer), pad_to_multiple_of=8)
    model.config.vocab_size = round_to_multiple(len(target_tokenizer), 8)

    target_input = model.get_input_embeddings().weight.data
    target_output_layer = model.get_output_embeddings()
    target_output = target_output_layer.weight.data if target_output_layer is not None else None

    for token_id in range(len(target_tokenizer)):
        token = target_tokenizer.convert_ids_to_tokens(token_id)
        source_ids = source_ids_for_target_token(source_tokenizer, token)
        source_ids = [idx for idx in source_ids if 0 <= idx < source_input.shape[0]]
        if not source_ids:
            continue
        source_tensor = torch.tensor(source_ids, device=source_input.device)
        target_input[token_id] = source_input[source_tensor].mean(dim=0).to(target_input.device)
        if source_output is not None and target_output is not None:
            source_tensor = source_tensor.to(source_output.device)
            target_output[token_id] = source_output[source_tensor].mean(dim=0).to(target_output.device)

    return model


def encode_text(tokenizer, text: str, add_special_tokens: bool) -> list[int]:
    return tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        verbose=False,
    )["input_ids"]


def simple_logical_chunking(examples: dict[str, list[str]], tokenizer, max_tokens: int):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    has_bos = tokenizer.bos_token_id is not None
    has_eos = tokenizer.eos_token_id is not None
    special_count = (1 if has_bos else 0) + (1 if has_eos else 0)
    payload_size = max_tokens - special_count
    if payload_size <= 0:
        raise ValueError(f"max_tokens={max_tokens} is too small for tokenizer special tokens")

    def add_special_tokens(token_ids: list[int]) -> list[int]:
        result = list(token_ids)
        if has_bos:
            result = [tokenizer.bos_token_id] + result
        if has_eos:
            result = result + [tokenizer.eos_token_id]
        return result

    chunks: list[list[int]] = []

    def append_chunk(payload_ids: list[int]) -> None:
        if not payload_ids:
            return
        chunk = add_special_tokens(payload_ids[:payload_size])
        chunks.append(chunk + [pad_id] * (max_tokens - len(chunk)))

    for text in examples["text"]:
        text = (text or "").strip()
        if not text:
            continue

        current_ids: list[int] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if not sentence:
                continue

            ids = encode_text(tokenizer, sentence, add_special_tokens=False)
            if not ids:
                continue

            if current_ids:
                ids_with_space = encode_text(tokenizer, " " + sentence, add_special_tokens=False)
                if len(current_ids) + len(ids_with_space) <= payload_size:
                    current_ids.extend(ids_with_space)
                    continue
                append_chunk(current_ids)
                current_ids = []

            if len(ids) > payload_size:
                for start in range(0, len(ids), payload_size):
                    append_chunk(ids[start : start + payload_size])
            else:
                current_ids = ids

        append_chunk(current_ids)

    return {"input_ids": chunks}


def load_text_datasets(train_path: str, val_path: str | None):
    train_dataset = datasets.load_dataset(
        "text",
        data_files={"train": train_path},
        split="train",
        cache_dir="./",
    )
    val_dataset = None
    if val_path:
        val_dataset = datasets.load_dataset(
            "text",
            data_files={"validation": val_path},
            split="validation",
            cache_dir="./",
        )
    return train_dataset, val_dataset


def tokenize_datasets(train_dataset, val_dataset, tokenizer, max_length: int, num_proc: int):
    train_tokenized = train_dataset.map(
        lambda examples: simple_logical_chunking(examples, tokenizer, max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
    )
    val_tokenized = None
    if val_dataset is not None:
        val_tokenized = val_dataset.map(
            lambda examples: simple_logical_chunking(examples, tokenizer, max_length),
            batched=True,
            num_proc=num_proc,
            remove_columns=val_dataset.column_names,
        )
    return train_tokenized, val_tokenized


def file_signature(path: str | Path) -> dict[str, object]:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def tokenizer_signature(tokenizer_path: str | Path) -> dict[str, object]:
    path = Path(tokenizer_path).expanduser().resolve()
    if path.is_file():
        return {"path": str(path), "files": [file_signature(path)]}

    files = []
    for name in [
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        candidate = path / name
        if candidate.exists():
            files.append(file_signature(candidate))
    return {"path": str(path), "files": files}


def tokenized_cache_key(
    train_path: str | Path,
    val_path: str | Path | None,
    tokenizer_path: str | Path,
    max_length: int,
) -> tuple[str, dict[str, object]]:
    metadata = {
        "chunking_version": CHUNKING_VERSION,
        "train_dataset": file_signature(train_path),
        "val_dataset": file_signature(val_path) if val_path else None,
        "tokenizer": tokenizer_signature(tokenizer_path),
        "max_length": max_length,
    }
    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], metadata


def load_or_tokenize_datasets(
    train_path: str | Path,
    val_path: str | Path | None,
    tokenizer,
    tokenizer_path: str | Path,
    max_length: int,
    num_proc: int,
    cache_root: str | Path,
):
    cache_key, metadata = tokenized_cache_key(train_path, val_path, tokenizer_path, max_length)
    cache_dir = Path(cache_root) / cache_key
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "validation"
    metadata_path = cache_dir / "metadata.json"

    has_cache = train_cache_dir.exists() and (val_path is None or val_cache_dir.exists())
    if has_cache:
        try:
            train_tokenized = datasets.load_from_disk(str(train_cache_dir))
            val_tokenized = datasets.load_from_disk(str(val_cache_dir)) if val_path else None
            metadata.update(
                {
                    "cache_key": cache_key,
                    "cache_dir": str(cache_dir),
                    "cache_hit": True,
                    "train_num_rows": len(train_tokenized),
                    "val_num_rows": len(val_tokenized) if val_tokenized is not None else 0,
                }
            )
            return train_tokenized, val_tokenized, metadata
        except Exception as exc:
            print(f"tokenized cache load failed, rebuilding: {exc}")

    train_dataset, val_dataset = load_text_datasets(str(train_path), str(val_path) if val_path else None)
    train_tokenized, val_tokenized = tokenize_datasets(
        train_dataset,
        val_dataset,
        tokenizer,
        max_length,
        num_proc,
    )

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_tokenized.save_to_disk(str(train_cache_dir))
    if val_tokenized is not None:
        val_tokenized.save_to_disk(str(val_cache_dir))

    metadata.update(
        {
            "cache_key": cache_key,
            "cache_dir": str(cache_dir),
            "cache_hit": False,
            "train_num_rows": len(train_tokenized),
            "val_num_rows": len(val_tokenized) if val_tokenized is not None else 0,
        }
    )
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return train_tokenized, val_tokenized, metadata


def read_selected_tokens(path: str | Path) -> list[str]:
    tokens = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            token = line.rstrip("\n").split("\t", 1)[0]
            if line_no == 0 and token == "token":
                continue
            if token:
                tokens.append(token)
    return tokens


def count_split_token_hits(dataset, target_ids: set[int], batch_size: int = 1000) -> tuple[Counter, int, int]:
    counts: Counter = Counter()
    sample_count = 0
    samples_with_hits = 0
    for batch in dataset.iter(batch_size=batch_size):
        for input_ids in batch["input_ids"]:
            sample_count += 1
            has_hit = False
            for token_id in input_ids:
                if token_id in target_ids:
                    counts[token_id] += 1
                    has_hit = True
            if has_hit:
                samples_with_hits += 1
    return counts, sample_count, samples_with_hits


def write_selected_token_hit_debug(
    output_dir: str | Path,
    tokenizer,
    train_tokenized,
    selected_tokens_path: str | Path,
) -> dict[str, object]:
    selected_tokens = read_selected_tokens(selected_tokens_path)
    vocab = tokenizer.get_vocab()
    token_rows = []
    target_ids = set()
    for rank, token in enumerate(selected_tokens, start=1):
        token_id = vocab.get(token)
        if token_id is not None:
            target_ids.add(token_id)
        token_rows.append({"rank": rank, "token": token, "token_id": token_id})

    counts, train_samples, train_samples_with_hits = count_split_token_hits(train_tokenized, target_ids)
    rows = []
    for row in token_rows:
        token_id = row["token_id"]
        train_count = counts[token_id] if token_id is not None else 0
        rows.append(
            {
                "rank": row["rank"],
                "token": row["token"],
                "token_id": token_id,
                "in_tokenizer": token_id is not None,
                "train_count": train_count,
            }
        )

    zero_hit_count = sum(1 for row in rows if row["train_count"] == 0)
    missing_from_tokenizer_count = sum(1 for row in rows if not row["in_tokenizer"])
    top20 = sorted(rows, key=lambda row: (-row["train_count"], row["rank"]))[:20]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = output_dir / "debug_selected_token_hits.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("rank\ttoken\ttoken_id\tin_tokenizer\ttrain_count\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row["rank"]),
                        row["token"],
                        "" if row["token_id"] is None else str(row["token_id"]),
                        str(row["in_tokenizer"]),
                        str(row["train_count"]),
                    ]
                )
                + "\n"
            )

    summary = {
        "selected_tokens_path": str(Path(selected_tokens_path).expanduser().resolve()),
        "selected_token_count": len(rows),
        "tokens_in_tokenizer_count": len(rows) - missing_from_tokenizer_count,
        "missing_from_tokenizer_count": missing_from_tokenizer_count,
        "zero_hit_count": zero_hit_count,
        "hit_token_count": len(rows) - zero_hit_count,
        "train_total_hits": sum(counts.values()),
        "train_samples": train_samples,
        "train_samples_with_hits": train_samples_with_hits,
        "top20": top20,
        "tsv_path": str(tsv_path),
    }
    json_path = output_dir / "debug_selected_token_hits.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["json_path"] = str(json_path)
    return summary


def load_model(model_path: str, dtype=torch.bfloat16):
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")


def apply_lora(model, args: Any, save_embedding_modules: bool):
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    modules_to_save = ["lm_head", "embed_tokens"] if save_embedding_modules else None
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        modules_to_save=modules_to_save,
    )
    return get_peft_model(model, config)


def data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            # print("Trainable:", name, "||", param.numel())
            trainable += param.numel()
    pct = 100 * trainable / total if total else 0
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct}")
