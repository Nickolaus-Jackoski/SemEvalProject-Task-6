import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score


# -----------------------------
# Task 1 only: direct_clarity
# -----------------------------
LABEL_COL = "clarity_label"
ID_COL = "index"
LABEL2ID = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_ORDER = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def find_project_root() -> Path:
    return Path(file).resolve().parent


def resolve_paths(project_root: Path):
    models_path = Path(os.environ.get("LOCAL_MODELS_PATH", str(project_root / "models"))).resolve()
    results_path = Path(os.environ.get("LOCAL_RESULTS_PATH", str(project_root / "results"))).resolve()
    results_path.mkdir(parents=True, exist_ok=True)
    return models_path, results_path


def find_checkpoint(models_path: Path, project_root: Path, cwd: Path, ckpt_name: str) -> Path | None:
    for p in [models_path / ckpt_name, project_root / ckpt_name, cwd / ckpt_name]:
        if p.exists() and p.is_dir():
            return p
    return None


def tokenizer_load_path(model_name: str, ckpt_path: Path | None) -> str:
    if ckpt_path:
        # if tokenizer artifacts exist in checkpoint, use them
        for fname in [
            "tokenizer.json", "tokenizer_config.json", "vocab.json", "vocab.txt",
            "merges.txt", "sentencepiece.bpe.model", "special_tokens_map.json"
        ]:
            if (ckpt_path / fname).exists():
                return str(ckpt_path)
    return model_name


def run_inference(model, tokenizer, tokenized_ds, device):
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # try larger batch sizes first; auto-back off on OOM
    for bs in [128, 64, 32, 16]:
        try:
            dl = DataLoader(tokenized_ds, batch_size=bs, shuffle=False, collate_fn=collator)
            preds = []
            model.eval()
            with torch.no_grad():
                for batch in tqdm(dl, desc=f"Inference (bs={bs})"):
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            return preds
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    raise RuntimeError("OOM for all attempted batch sizes: 128, 64, 32, 16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()

    if args.experiment != "direct_clarity":
        raise ValueError("This encoder_inference.py is Task 1 only: --experiment must be 'direct_clarity'.")

    model_name = args.model_name
    model_id = model_name.split("/")[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    project_root = find_project_root()
    models_path, results_path = resolve_paths(project_root)

    ckpt_name = f"{model_id}-qaevasion-direct_clarity"
    ckpt_path = find_checkpoint(models_path, project_root, Path.cwd(), ckpt_name)

    if ckpt_path:
        print(f"Loading FINETUNED checkpoint: {ckpt_path}")
        model_load = str(ckpt_path)
    else:
        print(f"WARNING: No checkpoint folder found for {ckpt_name}. Falling back to base model: {model_name}")
        model_load = model_name

    tok_load = tokenizer_load_path(model_name, ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_load)
    # IMPORTANT: do NOT pass num_labels here; it can re-init the head and ruin the checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(model_load)
    model.to(device)

    max_len = 512

    ds = load_dataset("ailsntua/QEvasion")["test"]
    ds = ds.filter(lambda x: x.get(LABEL_COL, "") != "")

    if ID_COL not in ds.column_names:
        raise ValueError(f"Expected ID column '{ID_COL}' not found. Available columns: {ds.column_names}")

    ids = ds[ID_COL]
    true_labels = ds[LABEL_COL]

    def tok_fn(examples):
        texts = [q + " " + a for q, a in zip(examples["interview_question"], examples["interview_answer"])]
        return tokenizer(texts, truncation=True, max_length=max_len)

    # Keep only tokenized tensors to avoid collator trying to tensorize strings
    tokenized = ds.map(tok_fn, batched=True, num_proc=4, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    pred_ids = run_inference(model, tokenizer, tokenized, device)
    pred_labels = [ID2LABEL[i] for i in pred_ids]

    out_df = pd.DataFrame({
        "ID": ids,                 # submission-friendly name
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    })

    out_path = results_path / f"{model_id}-direct_clarity.csv"
    out_df.to_csv(out_path, index=False)

    mf1 = f1_score(true_labels, pred_labels, average="macro", labels=LABEL_ORDER)
    print(f"Saved: {out_path} (rows={len(out_df)}), Macro-F1={mf1:.4f}")


if name == "main":
    main()