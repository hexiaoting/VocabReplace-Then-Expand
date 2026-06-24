# S3 Evaluation

This stage currently prepares the medical-domain MultiClinSum clinical-report
one-shot evaluation inputs.

## Medical Data Preparation

The script reads only:

```text
/home/wenting/workspace/dataset/multiclinsum/multiclinsum_test_en/
```

It pairs `fulltext/*.txt` as source documents (`SD`) with
`summaries/*_sum.txt` as references (`RS`). It does not read
`multiclinsum_gs_train_en` or `multiclinsum_large-scale_train_en`.

Default OOV scoring uses the `Qwen2.5-7B` tokenizer from `config.py`. For each
unigram, the script checks whether the tokenizer splits it into more than one
subword. The OOV concentration is the fraction of fragmented unigram
occurrences.

Splits:

- `highOOV_SD`: top 10% by source-document OOV concentration as test, remaining
  90% as train.
- `highOOV_RS`: top 10% by reference-summary OOV concentration as test,
  remaining 90% as train.
- `random`: random 10% as test, remaining 90% as train.

For every test case, the script retrieves one most similar example from that
scenario's train set and formats a one-shot prompt.

Run:

```bash
python S3_evaluation/prepare_medical_dataset.py
```

Outputs default to `outputs_<model_slug>/inference_data/medical/`. With the default
`--tokenizer Qwen2.5-7B`, this is:

```text
outputs_qwen25-7b/inference_data/medical/ClinReport_1_highOOV_SD.json
outputs_qwen25-7b/inference_data/medical/ClinReport_1_highOOV_RS.json
outputs_qwen25-7b/inference_data/medical/ClinReport_1_random.json
```

Each JSON file is an array of records with exactly:

```json
{
  "icl_input": "...",
  "target": "..."
}
```

By default, one-shot retrieval uses local PubMedBERT embeddings with cosine
similarity:

```text
RETRIEVER_MODEL_PATH_MAP["medical_pubmedbert"]
# /home/wenting/workspace/models/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb
```

Override it with another local Hugging Face model path if needed:

```bash
python S3_evaluation/prepare_medical_dataset.py \
  --retriever_model_path /path/to/pubmedbert
```

For a dependency-light smoke test, use lexical Jaccard retrieval:

```bash
python S3_evaluation/prepare_medical_dataset.py --retriever_backend token_jaccard
```

Switch the OOV tokenizer:

```bash
python S3_evaluation/prepare_medical_dataset.py --tokenizer Llama3.1-8B
```


## Medical Inference And Evaluation

Run BASE inference and evaluation:

```bash
python S3_evaluation/inference_and_evaluate.py \
  --method base \
  --tokenizer Qwen2.5-7B
```

Run CPTOnly or VOCABADAPT with an explicit LoRA adapter checkpoint:

```bash
python S3_evaluation/inference_and_evaluate.py \
  --method cptonly \
  --tokenizer Qwen2.5-7B \
  --adapter_path /path/to/cptonly/adapter

python S3_evaluation/inference_and_evaluate.py \
  --method vocabadapt \
  --tokenizer Qwen2.5-7B \
  --adapter_path /path/to/vocabadapt/adapter
```

Default input:

```text
outputs_<model_slug>/inference_data/medical/
```

Default output:

```text
outputs_<model_slug>/evaluation_medical/<method>/
```

The script uses greedy decoding (`do_sample=false`, `num_beams=1`). The paper
specifies a 100-word medical summary instruction in the prompt but does not
report a `max_new_tokens` value, so the script defaults to `--max_new_tokens 100`
as a conservative generation cap.

Metrics:

- `rouge_lcs`
- `bertscore` with `BERTSCORE_MODEL_PATH_MAP["medical_biobert"]`
- `frsr_sd` and `frsr_rs` only for Fragment Score
- `novel_unigram_concentration` on generated summaries

Recompute metrics from saved predictions without rerunning inference:

```bash
python S3_evaluation/inference_and_evaluate.py \
  --mode evaluate \
  --method base \
  --tokenizer Qwen2.5-7B
```
