# S2 Train

This stage supports two training methods:

- `cptonly`: continual pretraining with the original base tokenizer.
- `vocabadapt`: build/use a replacement-then-expansion tokenizer, then run the same CPT training flow.

All model-specific outputs live at the repository root:

```text
outputs_qwen25-7b/
outputs_llama31-8b/
```

## 1. Build VOCABADAPT Tokenizer

Use the selected medical tokens from S1 and the repository replacement-id files.

```bash
python S2_train/build_vocabadapt_tokenizer.py --tokenizer Qwen2.5-7B
python S2_train/build_vocabadapt_tokenizer.py --tokenizer Llama3.1-8B
```

Default selected token inputs are inferred from the tokenizer name:

```text
outputs_qwen25-7b/selected_tokens/top_10000_tokens_with_freq.tsv
outputs_llama31-8b/selected_tokens/top_10000_tokens_with_freq.tsv
```

Default tokenizer outputs:

```text
outputs_qwen25-7b/vocabadapt/tokenizer/
outputs_llama31-8b/vocabadapt/tokenizer/
```

Replacement is enabled by default, matching the paper's `Magikarp + Unreachable` setup:

```bash
python S2_train/build_vocabadapt_tokenizer.py \
  --tokenizer Qwen2.5-7B \
  --replace_unreachable_tokens true
```

To build an expansion-only tokenizer with the same selected tokens:

```bash
python S2_train/build_vocabadapt_tokenizer.py \
  --tokenizer Qwen2.5-7B \
  --replace_unreachable_tokens false
```

The tokenizer build writes:

```text
build_config.json
build_summary.json
replaced_tokens.pkl
replaced_tokens.tsv
```

## 2. Train CPTOnly

```bash
python S2_train/train.py --method cptonly --tokenizer Qwen2.5-7B
python S2_train/train.py --method cptonly --tokenizer Llama3.1-8B
```

Default data:

```text
train: data/pretraining-data/100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt
valid: data/pretraining-data/1MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Valid.txt
```

Default checkpoint roots:

```text
outputs_qwen25-7b/cptonly/checkpoints/<run_id>/
outputs_llama31-8b/cptonly/checkpoints/<run_id>/
```

## 3. Train VOCABADAPT

Build the tokenizer first, then train:

```bash
python S2_train/train.py --method vocabadapt --tokenizer Qwen2.5-7B
python S2_train/train.py --method vocabadapt --tokenizer Llama3.1-8B
```

Default tokenizer inputs:

```text
outputs_qwen25-7b/vocabadapt/tokenizer/
outputs_llama31-8b/vocabadapt/tokenizer/
```

Default checkpoint roots:

```text
outputs_qwen25-7b/vocabadapt/checkpoints/<run_id>/
outputs_llama31-8b/vocabadapt/checkpoints/<run_id>/
```

`<run_id>` is a UTC timestamp like `20260624T091530Z`. Passing
`--output_dir` overrides this and uses the exact path provided.

## Device Selection

Use `--cuda_visible_devices` to select the physical GPUs visible to the run:

```bash
python S2_train/train.py --method vocabadapt --tokenizer Qwen2.5-7B --cuda_visible_devices 0,2
```

The script sets `CUDA_VISIBLE_DEVICES` before importing PyTorch. With
`--cuda_visible_devices 0,2`, Hugging Face `device_map="auto"` only sees
physical GPUs 0 and 2; inside the process they are exposed as logical
`cuda:0` and `cuda:1`.

## Tokenized Dataset Cache

Training automatically caches tokenized datasets under the method directory:

```text
outputs_<model>/<method>/tokenized_cache/<cache_key>/
```

The cache key includes the train/validation file signatures, tokenizer files,
`max_length`, and the chunking implementation version. A cache hit skips
`tokenize_datasets`; the run prints `tokenized_cache_hit: True` and records
the cache metadata in `train_config.json`.

## Debug Selected Token Hits

For VOCABADAPT, enable debug statistics with:

```bash
python S2_train/train.py --method vocabadapt --tokenizer Qwen2.5-7B --debug true
```

This reads the S1 selected token file and counts each selected token's actual
ID occurrences in `train_tokenized`. By default the selected token path is read
from the tokenizer `build_summary.json`, falling back to:

```text
outputs_<model>/selected_tokens/top_10000_tokens_with_freq.tsv
```

You can override it with `--selected_tokens_path`. Outputs:

```text
outputs_<model>/vocabadapt/debug/debug_selected_token_hits.tsv
outputs_<model>/vocabadapt/debug/debug_selected_token_hits.json
```

The JSON summary records `zero_hit_count` and the top 20 most-hit selected
tokens. `--debug` is ignored for `cptonly`.

## Default Training Parameters

Defaults follow `paper.md` unless noted:

```text
LoRA r: 32
LoRA alpha: 64
learning_rate: 2e-5
num_train_epochs: 3
effective_batch_size: 64
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
max_length: 512
bf16: true
tf32: true
gradient_checkpointing: true
LoRA: always enabled
modules_to_save: vocabadapt -> lm_head, embed_tokens; cptonly -> none
```

## Run Records

Each training run writes:

```text
outputs_<model>/<method>/checkpoints/<run_id>/train_config.json
outputs_<model>/<method>/train_config.latest.json
outputs_<model>/runs.tsv
```

If an older `runs.tsv` with an incompatible header already exists, new rows are
written to `runs_v2.tsv` instead.

The per-run `train_config.json` records resolved paths, device visibility,
hyperparameters, LoRA settings, embedding module save policy, final vocabulary
size, and tokenizer build summary when available. `train_config.latest.json` is
only a convenience pointer to the most recent run. `runs.tsv` contains one
compact row per run for comparison.
