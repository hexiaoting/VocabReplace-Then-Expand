from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

MODELS_DIR = Path("/home/wenting/workspace/models")

MODEL_PATH_MAP = {
    "Llama-3-8B": MODELS_DIR / "meta-llama__Meta-Llama-3-8B",
    "Llama3-8B": MODELS_DIR / "meta-llama__Meta-Llama-3-8B",
    "Llama3.1-8B": MODELS_DIR / "meta-llama__Llama-3.1-8B",
    "Llama3.1-8B-instruct": MODELS_DIR / "meta-llama__Llama-3.1-8B-Instruct",
    "Llama3-8B-instruct": MODELS_DIR / "meta-llama__Meta-Llama-3-8B-Instruct",
    "Llama3.2-3B": MODELS_DIR / "meta-llama__Llama-3.2-3B",
    "Llama3.2-3B-instruct": MODELS_DIR / "meta-llama__Llama-3.2-3B-Instruct",
    "Mistral-7B": MODELS_DIR / "mistralai__Mistral-7B-v0.1",
    "Qwen2.5-7B": MODELS_DIR / "Qwen2.5-7B",
    "Qwen3-0.7B": MODELS_DIR / "Qwen3-0.7B",
}


RETRIEVER_MODEL_PATH_MAP = {
    "medical_pubmedbert": MODELS_DIR / "PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb",
}

BERTSCORE_MODEL_PATH_MAP = {
    "medical_biobert": MODELS_DIR / "biobert-base-cased-v1.1",
}

VACABULARITY_SIZE = 10000

LORA_TRAINING_PARAMS = {
    "r": 32 ,
    "lora_alpha": 64, 
    "learning_rate": 2e-5,
    "train_epochs": 3,
    "train_batch_size": 64 
}
