#!/usr/bin/env python
"""Run medical-domain inference and evaluation for prepared MultiClinSum prompts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "S2_train"))

from config import BERTSCORE_MODEL_PATH_MAP, MODEL_PATH_MAP  # noqa: E402


SCENARIO_FILES = {
    "highOOV_SD": "ClinReport_1_highOOV_SD.json",
    "highOOV_RS": "ClinReport_1_highOOV_RS.json",
    "random": "ClinReport_1_random.json",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SD_START = "Clinical Case Report 2:"
SD_END = "Discharge Summary 2:"
STOP_MARKERS = ["\n##", "\nClinical Case Report", "\nDischarge Summary"]
DEFAULT_METRICS = ["rouge_lcs", "bertscore", "fragment", "novel_unigram"]


def str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(".", "")


def default_input_dir(model_name: str) -> Path:
    return REPO_ROOT / f"outputs_{model_slug(model_name)}" / "inference_data" / "medical"


def default_output_dir(model_name: str, method: str) -> Path:
    return REPO_ROOT / f"outputs_{model_slug(model_name)}" / "evaluation_medical" / method


def default_tokenizer_path(method: str, model_name: str) -> Path:
    if method == "vocabadapt":
        return REPO_ROOT / f"outputs_{model_slug(model_name)}" / "vocabadapt" / "tokenizer"
    return Path(MODEL_PATH_MAP[model_name])


def dtype_from_arg(dtype: str):
    import torch

    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["infer", "evaluate", "infer_and_evaluate"], default="infer_and_evaluate")
    parser.add_argument("--method", required=True, choices=["base", "cptonly", "vocabadapt"])
    parser.add_argument("--tokenizer", default="Qwen2.5-7B", choices=sorted(MODEL_PATH_MAP))
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIO_FILES), choices=sorted(SCENARIO_FILES))

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_input_tokens", type=int, default=0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--trust_remote_code", type=str_to_bool, default=False)
    parser.add_argument("--trim_stop_markers", type=str_to_bool, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-scenario sample cap for smoke tests.")

    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=DEFAULT_METRICS)
    parser.add_argument("--bertscore_model_path", default=str(BERTSCORE_MODEL_PATH_MAP["medical_biobert"]))
    parser.add_argument("--bertscore_batch_size", type=int, default=8)
    parser.add_argument("--bertscore_device", default=None)
    parser.add_argument("--rouge_use_stemmer", type=str_to_bool, default=True)
    parser.add_argument("--fragment_prefix_space", type=str_to_bool, default=True)
    return parser.parse_args()


def read_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_scenario_inputs(input_dir: Path, scenarios: list[str], limit: int) -> dict[str, list[dict[str, Any]]]:
    data = {}
    for scenario in scenarios:
        path = input_dir / SCENARIO_FILES[scenario]
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared input for {scenario}: {path}")
        rows = read_json(path)
        for index, row in enumerate(rows):
            if "icl_input" not in row or "target" not in row:
                raise ValueError(f"{path} row {index} must contain icl_input and target")
        data[scenario] = rows[:limit] if limit and limit > 0 else rows
    return data


def prediction_path(output_dir: Path, scenario: str) -> Path:
    stem = Path(SCENARIO_FILES[scenario]).stem
    return output_dir / f"{stem}_predictions.json"


def metrics_path(output_dir: Path, scenario: str) -> Path:
    stem = Path(SCENARIO_FILES[scenario]).stem
    return output_dir / f"{stem}_metrics.json"


def extract_source_document(icl_input: str) -> str:
    start = icl_input.rfind(SD_START)
    if start < 0:
        raise ValueError("Could not find Clinical Case Report 2 marker in icl_input")
    start += len(SD_START)
    end = icl_input.find(SD_END, start)
    if end < 0:
        raise ValueError("Could not find Discharge Summary 2 marker in icl_input")
    return icl_input[start:end].strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def clean_prediction(text: str, trim_stop_markers: bool) -> str:
    text = (text or "").strip()
    if trim_stop_markers:
        positions = [text.find(marker) for marker in STOP_MARKERS if text.find(marker) >= 0]
        if positions:
            text = text[: min(positions)].strip()
    return text


def load_tokenizer(tokenizer_path: Path, trust_remote_code: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_generation_model(args: argparse.Namespace, tokenizer):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model_path = Path(args.base_model_path).expanduser().resolve() if args.base_model_path else Path(MODEL_PATH_MAP[args.tokenizer]).expanduser().resolve()
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if args.method in {"cptonly", "vocabadapt"} and not args.adapter_path:
        raise ValueError(f"--adapter_path is required for --method {args.method} when running inference")

    print(f"loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype_from_arg(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    if args.method == "vocabadapt":
        from training_utils import initialize_embeddings_by_mean

        print("initializing/resizing VOCABADAPT embeddings before adapter load")
        source_tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=args.trust_remote_code)
        model = initialize_embeddings_by_mean(model, source_tokenizer, tokenizer)

    if args.adapter_path:
        adapter_path = Path(args.adapter_path).expanduser().resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        print(f"loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return model


def model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def generate_batch(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    import torch

    encode_kwargs = {"padding": True, "return_tensors": "pt"}
    if args.max_input_tokens and args.max_input_tokens > 0:
        encode_kwargs.update({"truncation": True, "max_length": args.max_input_tokens})
    encoded = tokenizer(prompts, **encode_kwargs)
    device = model_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_width = encoded["input_ids"].shape[1]
    predictions = []
    for row_ids in output_ids:
        new_ids = row_ids[prompt_width:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        predictions.append(clean_prediction(raw, args.trim_stop_markers))
    return predictions


def run_inference(
    scenario_inputs: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    model,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    all_predictions = {}
    for scenario, rows in scenario_inputs.items():
        print(f"running inference: {scenario} ({len(rows)} samples)")
        predictions = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            batch_predictions = generate_batch(model, tokenizer, [row["icl_input"] for row in batch], args)
            for row, prediction in zip(batch, batch_predictions):
                predictions.append({**row, "prediction": prediction})
            print(f"{scenario}: generated {min(start + args.batch_size, len(rows))}/{len(rows)}")
        all_predictions[scenario] = predictions
        path = prediction_path(output_dir, scenario)
        write_json(path, predictions)
        print(f"saved predictions: {path}")
    return all_predictions


def load_predictions(output_dir: Path, scenarios: list[str]) -> dict[str, list[dict[str, Any]]]:
    data = {}
    for scenario in scenarios:
        path = prediction_path(output_dir, scenario)
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {scenario}: {path}")
        rows = read_json(path)
        for index, row in enumerate(rows):
            if "icl_input" not in row or "target" not in row or "prediction" not in row:
                raise ValueError(f"{path} row {index} must contain icl_input, target, and prediction")
        data[scenario] = rows
    return data


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def fallback_rouge_lcs(prediction: str, reference: str) -> float:
    pred_tokens = [token.lower() for token in words(prediction)]
    ref_tokens = [token.lower() for token in words(reference)]
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def rouge_lcs_scores(predictions: list[str], references: list[str], use_stemmer: bool) -> list[float]:
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return [fallback_rouge_lcs(pred, ref) for pred, ref in zip(predictions, references)]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def bertscore_scores(predictions: list[str], references: list[str], args: argparse.Namespace) -> dict[str, list[float]]:
    from bert_score import score as bert_score

    device = args.bertscore_device
    precision, recall, f1 = bert_score(
        predictions,
        references,
        model_type=args.bertscore_model_path,
        batch_size=args.bertscore_batch_size,
        lang="en",
        device=device,
        verbose=True,
        rescale_with_baseline=False,
    )
    return {
        "bertscore_precision": [float(value) for value in precision.tolist()],
        "bertscore_recall": [float(value) for value in recall.tolist()],
        "bertscore_f1": [float(value) for value in f1.tolist()],
    }


def fragment_score(text: str, tokenizer, prefix_space: bool) -> float:
    text_words = words(text)
    if not text_words:
        return 0.0
    total_subwords = 0
    for token in text_words:
        probe = f" {token}" if prefix_space else token
        total_subwords += len(tokenizer.encode(probe, add_special_tokens=False))
    return total_subwords / len(text_words)


def novel_unigram_concentration(prediction: str, source_document: str) -> float:
    source_words = {token.lower() for token in words(source_document)}
    pred_words = [token.lower() for token in words(prediction)]
    if not pred_words:
        return 0.0
    novel_count = sum(1 for token in pred_words if token not in source_words)
    return novel_count / len(pred_words)


def mean_or_zero(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def evaluate_scenario(
    rows: list[dict[str, Any]],
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, Any]:
    references = [str(row["target"]) for row in rows]
    predictions = [str(row["prediction"]) for row in rows]
    source_documents = [extract_source_document(str(row["icl_input"])) for row in rows]

    per_sample = []
    for row, source_document in zip(rows, source_documents):
        per_sample.append(
            {
                "icl_input": row["icl_input"],
                "target": row["target"],
                "prediction": row["prediction"],
                "source_document": source_document,
            }
        )

    if "rouge_lcs" in args.metrics:
        scores = rouge_lcs_scores(predictions, references, args.rouge_use_stemmer)
        for sample, score in zip(per_sample, scores):
            sample["rouge_lcs"] = score

    if "bertscore" in args.metrics:
        scores = bertscore_scores(predictions, references, args)
        for sample_index, sample in enumerate(per_sample):
            for key, values in scores.items():
                sample[key] = values[sample_index]

    if "fragment" in args.metrics:
        for sample, source_document, reference in zip(per_sample, source_documents, references):
            sample["frsr_sd"] = fragment_score(source_document, tokenizer, args.fragment_prefix_space)
            sample["frsr_rs"] = fragment_score(reference, tokenizer, args.fragment_prefix_space)

    if "novel_unigram" in args.metrics:
        for sample, prediction, source_document in zip(per_sample, predictions, source_documents):
            sample["novel_unigram_concentration"] = novel_unigram_concentration(prediction, source_document)

    metric_keys = [key for key in per_sample[0] if key not in {"icl_input", "target", "prediction", "source_document"}] if per_sample else []
    aggregate = {key: mean_or_zero([float(sample[key]) for sample in per_sample]) for key in metric_keys}
    aggregate["sample_count"] = len(per_sample)
    return {"aggregate": aggregate, "per_sample": per_sample}


def run_evaluation(
    predictions_by_scenario: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    summary = {}
    for scenario, rows in predictions_by_scenario.items():
        print(f"evaluating: {scenario} ({len(rows)} samples)")
        result = evaluate_scenario(rows, tokenizer, args)
        result["scenario"] = scenario
        result["method"] = args.method
        result["tokenizer"] = args.tokenizer
        path = metrics_path(output_dir, scenario)
        write_json(path, result)
        summary[scenario] = result["aggregate"]
        print(f"saved metrics: {path}")
    write_summary_files(output_dir, args, summary)
    return summary


def write_summary_files(output_dir: Path, args: argparse.Namespace, summary: dict[str, dict[str, Any]]) -> None:
    summary_json = {
        "method": args.method,
        "tokenizer": args.tokenizer,
        "metrics": args.metrics,
        "scenarios": summary,
    }
    write_json(output_dir / "metrics_summary.json", summary_json)

    keys = sorted({key for aggregate in summary.values() for key in aggregate})
    csv_path = output_dir / "metrics_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario"] + keys)
        writer.writeheader()
        for scenario, aggregate in summary.items():
            writer.writerow({"scenario": scenario, **aggregate})
    print(f"saved summary: {output_dir / 'metrics_summary.json'}")
    print(f"saved summary: {csv_path}")


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else default_input_dir(args.tokenizer)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.tokenizer, args.method)
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else default_tokenizer_path(args.method, args.tokenizer)

    print(f"mode: {args.mode}")
    print(f"method: {args.method}")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"tokenizer_path: {tokenizer_path}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path, args.trust_remote_code)

    predictions_by_scenario = None
    if args.mode in {"infer", "infer_and_evaluate"}:
        scenario_inputs = load_scenario_inputs(input_dir, args.scenarios, args.limit)
        model = load_generation_model(args, tokenizer)
        predictions_by_scenario = run_inference(scenario_inputs, output_dir, model, tokenizer, args)

    if args.mode in {"evaluate", "infer_and_evaluate"}:
        if predictions_by_scenario is None:
            predictions_by_scenario = load_predictions(output_dir, args.scenarios)
        run_evaluation(predictions_by_scenario, output_dir, tokenizer, args)


if __name__ == "__main__":
    main()
