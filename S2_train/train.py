#!/usr/bin/env python
"""Train CPTOnly or VOCABADAPT models on domain text."""

import argparse
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def set_cuda_visible_devices_from_argv() -> None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        value = None
        if arg == "--cuda_visible_devices" and index + 1 < len(sys.argv):
            value = sys.argv[index + 1]
        elif arg.startswith("--cuda_visible_devices="):
            value = arg.split("=", 1)[1]
        if value:
            os.environ["CUDA_VISIBLE_DEVICES"] = value
            return


set_cuda_visible_devices_from_argv()

import torch
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from config import MODEL_PATH_MAP  # noqa: E402
from training_utils import (  # noqa: E402
    apply_lora,
    data_collator,
    initialize_embeddings_by_mean,
    load_model,
    load_or_tokenize_datasets,
    print_trainable_parameters,
    write_selected_token_hit_debug,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEFAULT_TRAIN_PATH = (
    REPO_ROOT
    / "data"
    / "pretraining-data"
    / "100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt"
)
DEFAULT_VAL_PATH = (
    REPO_ROOT
    / "data"
    / "pretraining-data"
    / "1MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Valid.txt"
)


def str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "")


def model_output_root(model_name: str) -> Path:
    return REPO_ROOT / f"outputs_{model_slug(model_name)}"


def default_tokenizer_path(method: str, model_name: str) -> Path:
    if method == "cptonly":
        return Path(MODEL_PATH_MAP[model_name])
    return model_output_root(model_name) / "vocabadapt" / "tokenizer"


def default_selected_tokens_path(model_name: str) -> Path:
    return model_output_root(model_name) / "selected_tokens" / "top_10000_tokens_with_freq.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["cptonly", "vocabadapt"])
    parser.add_argument("--tokenizer", required=True, choices=sorted(MODEL_PATH_MAP))
    parser.add_argument("--dataset_path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val_dataset_path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--logging_dir", default=None)
    parser.add_argument("--debug", type=str_to_bool, default=False)
    parser.add_argument("--selected_tokens_path", default=None)
    parser.add_argument("--cuda_visible_devices", default=0)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bf16", type=str_to_bool, default=True)
    parser.add_argument("--tf32", type=str_to_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--overwrite_output_dir", type=str_to_bool, default=True)
    return parser.parse_args()


def make_training_args(args: argparse.Namespace, output_dir: Path, logging_dir: Path, has_eval: bool):
    kwargs = {
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "seed": args.seed,
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "prediction_loss_only": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "do_train": True,
        "do_eval": has_eval,
        "lr_scheduler_type": args.lr_scheduler_type,
        "disable_tqdm": False,
        "label_names": ["labels"],
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_steps": args.eval_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
        "save_only_model": True,
        "optim": args.optim,
    }

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

    filtered = {key: value for key, value in kwargs.items() if key in signature and value is not None}
    return TrainingArguments(**filtered)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_run_row(root: Path, row: dict) -> None:
    path = root / "runs.tsv"
    header = [
        "timestamp",
        "run_id",
        "method",
        "tokenizer",
        "learning_rate",
        "epochs",
        "effective_batch_size",
        "final_vocab_size",
        "cuda_visible_devices",
        "torch_cuda_device_count",
        "replace_unreachable_tokens",
        "output_dir",
    ]
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8") as f:
            existing_header = f.readline().rstrip("\n").split("\t")
        if existing_header != header:
            path = root / "runs_v2.tsv"
            exists = path.exists()

    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(row.get(key, "")) for key in header) + "\n")


def main() -> None:
    args = parse_args()
    run_dt = datetime.now(timezone.utc)
    timestamp = run_dt.isoformat()
    run_id = run_dt.strftime("%Y%m%dT%H%M%SZ")
    root = model_output_root(args.tokenizer)
    method_dir = root / args.method
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else method_dir / "checkpoints" / run_id
    logging_dir = Path(args.logging_dir).expanduser().resolve() if args.logging_dir else method_dir / "logs" / run_id
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else default_tokenizer_path(args.method, args.tokenizer)
    model_path = Path(MODEL_PATH_MAP[args.tokenizer]).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    val_dataset_path = Path(args.val_dataset_path).expanduser().resolve() if args.val_dataset_path else None

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if val_dataset_path and not val_dataset_path.exists():
        raise FileNotFoundError(f"Validation dataset path does not exist: {val_dataset_path}")

    build_summary_path = tokenizer_path / "build_summary.json"
    tokenizer_build_summary = None
    if build_summary_path.exists():
        tokenizer_build_summary = json.load(build_summary_path.open("r", encoding="utf-8"))

    if args.selected_tokens_path:
        selected_tokens_path = Path(args.selected_tokens_path).expanduser().resolve()
    elif tokenizer_build_summary and tokenizer_build_summary.get("selected_tokens_path"):
        selected_tokens_path = Path(tokenizer_build_summary["selected_tokens_path"]).expanduser().resolve()
    else:
        selected_tokens_path = default_selected_tokens_path(args.tokenizer)

    if args.debug and args.method == "vocabadapt" and not selected_tokens_path.exists():
        raise FileNotFoundError(f"Selected tokens path does not exist: {selected_tokens_path}")

    print(f"tokenizer_path: {tokenizer_path}")
    print(f"model_path: {model_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"val_dataset_path: {val_dataset_path}")
    print(f"output_dir: {output_dir}")
    print(f"logging_dir: {logging_dir}")
    print(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch_cuda_device_count: {torch.cuda.device_count()}")
    if args.debug and args.method == "vocabadapt":
        print(f"selected_tokens_path: {selected_tokens_path}")
    print(f"args: {args}")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    source_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_cache_root = method_dir / "tokenized_cache"
    train_tokenized, val_tokenized, tokenized_cache_info = load_or_tokenize_datasets(
        dataset_path,
        val_dataset_path,
        tokenizer,
        tokenizer_path,
        args.max_length,
        args.num_proc,
        tokenized_cache_root,
    )
    print(f"tokenized_cache_dir: {tokenized_cache_info['cache_dir']}")
    print(f"tokenized_cache_hit: {tokenized_cache_info['cache_hit']}")

    debug_summary = None
    if args.debug and args.method == "vocabadapt":
        debug_summary = write_selected_token_hit_debug(
            method_dir / "debug",
            tokenizer,
            train_tokenized,
            selected_tokens_path,
        )
        print(f"debug_selected_token_hits: {debug_summary}")
    elif args.debug:
        print("--debug only applies to --method vocabadapt; skipping new-token hit stats.")

    model = load_model(str(model_path), torch.bfloat16 if args.bf16 else torch.float16)
    if args.method == "vocabadapt":
        model = initialize_embeddings_by_mean(model, source_tokenizer, tokenizer)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    save_embedding_modules = args.method == "vocabadapt"
    model = apply_lora(model, args, save_embedding_modules=save_embedding_modules)

    print_trainable_parameters(model)

    training_args = make_training_args(args, output_dir, logging_dir, val_tokenized is not None)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_tokenized is not None else [],
    )

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps

    train_config = {
        "timestamp": timestamp,
        "run_id": run_id,
        "method": args.method,
        "tokenizer": args.tokenizer,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "dataset_path": str(dataset_path),
        "val_dataset_path": str(val_dataset_path) if val_dataset_path else None,
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "debug": args.debug,
        "selected_tokens_path": str(selected_tokens_path) if args.debug and args.method == "vocabadapt" else None,
        "max_length": args.max_length,
        "num_proc": args.num_proc,
        "tokenized_cache": tokenized_cache_info,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_lora": True,
        "save_embedding_modules": save_embedding_modules,
        "bf16": args.bf16,
        "tf32": args.tf32,
        "gradient_checkpointing": args.gradient_checkpointing,
        "final_vocab_size": len(tokenizer),
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "modules_to_save": ["lm_head", "embed_tokens"] if save_embedding_modules else [],
    }
    if tokenizer_build_summary is not None:
        train_config["tokenizer_build_summary"] = tokenizer_build_summary
    if debug_summary is not None:
        train_config["debug_selected_token_hits"] = debug_summary

    write_json(output_dir / "train_config.json", train_config)
    write_json(method_dir / "train_config.latest.json", train_config)
    append_run_row(
        root,
        {
            "timestamp": timestamp,
            "run_id": run_id,
            "method": args.method,
            "tokenizer": args.tokenizer,
            "learning_rate": args.learning_rate,
            "epochs": args.num_train_epochs,
            "effective_batch_size": effective_batch,
            "final_vocab_size": len(tokenizer),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "replace_unreachable_tokens": (
                tokenizer_build_summary.get("replace_unreachable_tokens")
                if tokenizer_build_summary is not None
                else ""
            ),
            "output_dir": output_dir,
        },
    )

    trainer.train()
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
