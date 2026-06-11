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

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
BENCHMARK_BATCHES = 30
EVAL_SAMPLE_SIZE = 2000
VOCAB_PATH = Path("/tmp/bert-base-uncased-vocab.txt")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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
    candidates = [
        OUTPUT_DIR / "for_bert",
        ROOT / "for_bert",
        ROOT.parent / "for_bert",
        ROOT.parent.parent / "for_bert",
    ]
    for d in candidates:
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
    scores = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        scores.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, pred),
                "f1": f1_score(y_true, pred, zero_division=0),
            }
        )
    df = pd.DataFrame(scores).sort_values(["f1", "accuracy"], ascending=False)
    return float(df.iloc[0]["threshold"])


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Missing tokenizer vocab: {VOCAB_PATH}")

    split_dir = find_split_dir()
    val_df = load_split(split_dir / "val.csv").head(EVAL_SAMPLE_SIZE)
    test_df = load_split(split_dir / "test.csv").head(EVAL_SAMPLE_SIZE)

    tokenizer = BertTokenizer(vocab_file=str(VOCAB_PATH), do_lower_case=True)
    val_loader = DataLoader(PairDataset(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairDataset(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
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

    checkpoint_path = next(
        (
            p
            for p in [
                MODEL_DIR / "bert_crossencoder.pt",
                ROOT / "bert_crossencoder.pt",
                ROOT.parent / "bert_crossencoder.pt",
            ]
            if p.exists()
        ),
        None,
    )
    if checkpoint_path is None:
        raise FileNotFoundError("Missing bert_crossencoder.pt")

    float_model = BertForSequenceClassification(config)
    float_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    float_model = float_model.to(device).eval()

    val_prob, val_y = run_eval(float_model, val_loader, device)
    threshold = best_threshold(val_y, val_prob)

    quant_model = torch.quantization.quantize_dynamic(
        replace_bert_embeddings_with_int8(float_model),
        {nn.Linear},
        dtype=torch.qint8,
    )
    quant_model_path = MODEL_DIR / "bert_crossencoder_int8.pt"
    torch.save(quant_model.state_dict(), quant_model_path)

    test_prob, test_y = run_eval(quant_model, test_loader, device)
    pred = (test_prob >= threshold).astype(int)
    inference_sec, inference_ms = benchmark_latency(quant_model, test_loader, device, BENCHMARK_BATCHES)

    row = {
        "Model": "BERT Cross-Encoder Int8",
        "Owner": "Yuchan",
        "Role": "Embedding tables and linear layers quantized to int8",
        "Official Threshold Type": "validation_f1_best_sampled_quantization",
        "Best Threshold": threshold,
        "Accuracy Threshold": threshold,
        "Train Samples": int(len(load_split(split_dir / "train.csv"))),
        "Validation Samples": int(len(val_df)),
        "Test Samples": int(len(test_df)),
        "Test Accuracy": float(accuracy_score(test_y, pred)),
        "Test F1": float(f1_score(test_y, pred, zero_division=0)),
        "Test Precision": float((pred & test_y).sum() / max(pred.sum(), 1)),
        "Test Recall": float((pred & test_y).sum() / max(test_y.sum(), 1)),
        "Test Log Loss": float(log_loss(test_y, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "Training Time sec": None,
        "Inference Time sec": float(inference_sec),
        "Inference ms/sample": float(inference_ms),
        "Model Size MB": float(model_size_mb(quant_model_path)),
        "Encoder Size MB": None,
        "Parameter Count": int(sum(p.numel() for p in quant_model.parameters())),
        "Encoder Parameter Count": None,
        "State Tensor Count": int(sum(t.numel() for t in quant_model.state_dict().values() if torch.is_tensor(t))),
        "Memory RSS MB": None,
        "Max Sequence Length": MAX_LENGTH,
        "Vocabulary Size": 30522,
        "Embedding Dim": 768,
        "LSTM Units": None,
        "Batch Size": BATCH_SIZE,
        "Epochs Run": 3,
        "GPU": "CPU only",
        "Benchmark Scope": f"sampled_{EVAL_SAMPLE_SIZE}",
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "bert_crossencoder_quantized_results.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(pd.DataFrame([row]).to_string(index=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
