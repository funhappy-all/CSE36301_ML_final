from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer


if torch.backends.quantized.engine == "none" and torch.backends.quantized.supported_engines:
    for engine in ["fbgemm", "qnnpack", "x86"]:
        if engine in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = engine
            break

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULT_DIR = OUTPUT_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"

MAX_LENGTH = 128
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128
BENCHMARK_BATCHES = 5
VOCAB_PATH = Path("/tmp/bert-base-uncased-vocab.txt")


class PairDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: BertTokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            row["question1"],
            row["question2"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(row["is_duplicate"]), dtype=torch.long)
        return item


class Int8Embedding(nn.Module):
    def __init__(self, qweight: torch.Tensor, scale: float, padding_idx: int | None = None):
        super().__init__()
        self.register_buffer("qweight", qweight.to(torch.int8))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self.padding_idx = padding_idx

    @classmethod
    def from_float(cls, embedding: nn.Embedding) -> "Int8Embedding":
        weight = embedding.weight.detach().cpu()
        max_abs = weight.abs().max().clamp(min=1e-8)
        scale = float(max_abs / 127.0)
        qweight = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        if embedding.padding_idx is not None:
            qweight[embedding.padding_idx].zero_()
        return cls(qweight, scale, embedding.padding_idx)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        selected = F.embedding(token_ids, self.qweight, padding_idx=self.padding_idx)
        return selected.float() * self.scale


def replace_bert_embeddings_with_int8(model: BertForSequenceClassification) -> BertForSequenceClassification:
    converted = copy.deepcopy(model).cpu().eval()
    embeddings = converted.bert.embeddings
    embeddings.word_embeddings = Int8Embedding.from_float(embeddings.word_embeddings)
    embeddings.position_embeddings = Int8Embedding.from_float(embeddings.position_embeddings)
    embeddings.token_type_embeddings = Int8Embedding.from_float(embeddings.token_type_embeddings)
    return converted


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" in df.columns and "is_duplicate" not in df.columns:
        df = df.rename(columns={"label": "is_duplicate"})
    df["question1"] = df["question1"].fillna("").astype(str)
    df["question2"] = df["question2"].fillna("").astype(str)
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    return df


def find_split_dir() -> Path:
    for d in [OUTPUT_DIR / "for_bert", ROOT / "for_bert", ROOT.parent / "for_bert", ROOT.parent.parent / "for_bert"]:
        if (d / "train.csv").exists():
            return d
    raise FileNotFoundError("Could not find for_bert split CSVs.")


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            labels.append(batch["labels"].numpy())
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels)


def benchmark_latency(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> tuple[float, float]:
    model.eval()
    total_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            _ = model(**batch)
            total_samples += batch["input_ids"].size(0)
    elapsed = time.perf_counter() - start
    return elapsed, (elapsed * 1000.0 / max(total_samples, 1))


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.arange(0.20, 0.81, 0.01)
    rows = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, pred),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    return float(pd.DataFrame(rows).sort_values(["f1", "accuracy"], ascending=False).iloc[0]["threshold"])


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def make_row(model_name: str, model: nn.Module, val_loader: DataLoader, test_loader: DataLoader, threshold: float, size_path: Path, scope: str) -> dict[str, object]:
    val_prob, val_y = run_eval(model, val_loader, torch.device("cpu"))
    test_prob, test_y = run_eval(model, test_loader, torch.device("cpu"))
    infer_sec, infer_ms = benchmark_latency(model, test_loader, torch.device("cpu"), BENCHMARK_BATCHES)
    pred = (test_prob >= threshold).astype(int)
    return {
        "Model": model_name,
        "Owner": "Yuchan",
        "Role": "Embedding tables and linear layers quantized to int8" if "Int8" in model_name else "Float BERT Cross-Encoder",
        "Official Threshold Type": f"validation_f1_best_{scope}",
        "Best Threshold": threshold,
        "Accuracy Threshold": threshold,
        "Train Samples": 323432,
        "Validation Samples": int(len(val_y)),
        "Test Samples": int(len(test_y)),
        "Test Accuracy": float(accuracy_score(test_y, pred)),
        "Test F1": float(f1_score(test_y, pred, zero_division=0)),
        "Test Precision": float((pred & test_y).sum() / max(pred.sum(), 1)),
        "Test Recall": float((pred & test_y).sum() / max(test_y.sum(), 1)),
        "Test Log Loss": float(log_loss(test_y, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "Training Time sec": None,
        "Inference Time sec": float(infer_sec),
        "Inference ms/sample": float(infer_ms),
        "Model Size MB": float(model_size_mb(size_path)),
        "Encoder Size MB": None,
        "Parameter Count": int(sum(p.numel() for p in model.parameters())),
        "Encoder Parameter Count": None,
        "State Tensor Count": int(sum(t.numel() for t in model.state_dict().values() if torch.is_tensor(t))),
        "Memory RSS MB": None,
        "Max Sequence Length": MAX_LENGTH,
        "Vocabulary Size": 30522,
        "Embedding Dim": 768,
        "LSTM Units": None,
        "Batch Size": BATCH_SIZE,
        "Epochs Run": 3,
        "GPU": "CPU only",
        "Benchmark Scope": scope,
    }


def main() -> None:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(VOCAB_PATH)
    split_dir = find_split_dir()
    train_df = load_split(split_dir / "train.csv")
    val_df = load_split(split_dir / "val.csv")
    test_df = load_split(split_dir / "test.csv")
    tokenizer = BertTokenizer(vocab_file=str(VOCAB_PATH), do_lower_case=True)
    val_loader = DataLoader(PairDataset(val_df, tokenizer, MAX_LENGTH), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairDataset(test_df, tokenizer, MAX_LENGTH), batch_size=EVAL_BATCH_SIZE, shuffle=False)

    checkpoint_candidates = [MODEL_DIR / "bert_crossencoder_float.pt", MODEL_DIR / "bert_crossencoder.pt", ROOT / "bert_crossencoder.pt", ROOT.parent / "bert_crossencoder.pt"]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError("Missing BERT float checkpoint.")

    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_labels=2,
    )

    float_model = BertForSequenceClassification(config)
    float_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    float_model = float_model.cpu().eval()
    val_prob, val_y = run_eval(float_model, val_loader, torch.device("cpu"))
    threshold = best_threshold(val_y, val_prob)
    float_size_path = MODEL_DIR / "bert_crossencoder_float_fair.pt"
    torch.save(float_model.state_dict(), float_size_path)

    int8_model = torch.quantization.quantize_dynamic(replace_bert_embeddings_with_int8(float_model), {nn.Linear}, dtype=torch.qint8)
    int8_size_path = MODEL_DIR / "bert_crossencoder_int8_fair.pt"
    torch.save(int8_model.state_dict(), int8_size_path)

    rows = [
        make_row("BERT Cross-Encoder", float_model, val_loader, test_loader, threshold, float_size_path, "full_test_5batch_latency"),
        make_row("BERT Cross-Encoder Int8", int8_model, val_loader, test_loader, threshold, int8_size_path, "full_test_5batch_latency"),
    ]

    out_path = RESULT_DIR / "bert_crossencoder_quantized_results_fair.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows)[["Model", "Benchmark Scope", "Validation Samples", "Test Samples", "Test Accuracy", "Test F1", "Inference ms/sample", "Model Size MB"]].to_string(index=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
