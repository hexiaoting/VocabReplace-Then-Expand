# S1 Get Token

## 1. Train Medical BPE

Train the medical-domain ByteLevel BPE tokenizer.

Defaults follow `paper.md`:

- training data: `data/pretraining-data/100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt`
- BPE vocabulary size: `256000`

Run with defaults:

```bash
python S1_getToken/GenerateVocabulary.py
```

Override the corpus or vocabulary size:

```bash
python S1_getToken/GenerateVocabulary.py \
  --input_path data/pretraining-data/100MTokens-ClinicalGuidelines-PubMedArticles-PubMedAbstracts_Train.txt \
  --vocab_size 256000
```

The default output is:

```text
S1_getToken/VocabFiles-medical/bpe_256000/
```

## 2. Select Top Medical Tokens

Select the top medical-domain tokens for one base tokenizer. The tokenizer name
must exist in `MODEL_PATH_MAP` in `config.py`.

```bash
python S1_getToken/select_tokens.py --tokenizer Qwen2.5-7B
python S1_getToken/select_tokens.py --tokenizer Llama3.1-8B
```

Outputs:

```text
outputs_qwen25-7b/selected_tokens/top_10000_tokens_with_freq.tsv
outputs_llama31-8b/selected_tokens/top_10000_tokens_with_freq.tsv
```

The TSV columns are `token`, `token_id`, and `frequency`. Later tokenizer
construction steps should read the `token` column.
