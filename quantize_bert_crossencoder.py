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


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULT_DIR = OUTPUT_DIR / "results"
MODEL_DIR = OUTPUT_DIR / "models"
BERT_DIR = OUTPUT_DIR / "bert"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
BENCHMARK_BATCHES = 30


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_dirs = [
        OUTPUT_DIR / "for_bert",
        ROOT / "for_bert",
        ROOT.parent / "for_bert",
        ROOT.parent.parent / "for_bert",
    ]
    data_dir = next((d for d in candidate_dirs if (d / "train.csv").exists()), None)
    if data_dir is None:
        raise FileNotFoundError("Could not find for_bert/train.csv in the current repo or parent outputs.")

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    for df in (train_df, val_df, test_df):
        if "label" in df.columns and "is_duplicate" not in df.columns:
            df.rename(columns={"label": "is_duplicate"}, inplace=True)
        df["question1"] = df["question1"].fillna("").astype(str)
        df["question2"] = df["question2"].fillna("").astype(str)
        df["is_duplicate"] = df["is_duplicate"].astype(int)
    return train_df, val_df, test_df


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


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def build_row(
    model_name: str,
    model: nn.Module,
    loader_val: DataLoader,
    loader_test: DataLoader,
    device: torch.device,
    threshold: float,
    size_mb: float,
    parameter_count: int,
    inference_sec: float,
    inference_ms: float,
) -> dict[str, object]:
    val_prob, val_y = run_eval(model, loader_val, device)
    test_prob, test_y = run_eval(model, loader_test, device)
    pred = (test_prob >= threshold).astype(int)
    return {
        "Model": model_name,
        "Val Acc": float(accuracy_score(val_y, (val_prob >= threshold).astype(int))),
        "Test Acc": float(accuracy_score(test_y, pred)),
        "Test F1": float(f1_score(test_y, pred, zero_division=0)),
        "Log Loss": float(log_loss(test_y, np.clip(test_prob, 1e-6, 1 - 1e-6))),
        "Infer (ms/sample)": float(inference_ms),
        "Model Size (MB)": float(size_mb),
        "Parameter Count": parameter_count,
        "Inference Time sec": float(inference_sec),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cpu":
        print("Cross-encoder quantization is CPU-oriented; moving to CPU for benchmarking.")
        device = torch.device("cpu")

    _, val_df, test_df = load_data()
    vocab_candidates = [
        Path("/tmp/bert-base-uncased-vocab.txt"),
        ROOT / "bert-base-uncased-vocab.txt",
        ROOT.parent / "bert-base-uncased-vocab.txt",
    ]
    vocab_path = next((p for p in vocab_candidates if p.exists()), None)
    if vocab_path is None:
        raise FileNotFoundError(
            "Missing bert-base-uncased vocab.txt. Download it or place it at /tmp/bert-base-uncased-vocab.txt."
        )
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    val_loader = DataLoader(PairDataset(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairDataset(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    checkpoint_candidates = [
        MODEL_DIR / "bert_crossencoder.pt",
        ROOT / "bert_crossencoder.pt",
        ROOT.parent / "bert_crossencoder.pt",
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Missing checkpoint: bert_crossencoder.pt. Place it in the repo root or outputs/models/."
        )

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
    model = BertForSequenceClassification(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Use the float model's best validation threshold logic as the baseline.
    val_prob, val_y = run_eval(model, val_loader, device)
    thresholds = np.arange(0.20, 0.81, 0.01)
    scores = []
    for threshold in thresholds:
        pred = (val_prob >= threshold).astype(int)
        scores.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(val_y, pred),
                "f1": f1_score(val_y, pred, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(scores).sort_values(["f1", "accuracy"], ascending=False)
    best_threshold = float(threshold_df.iloc[0]["threshold"])

    float_model = copy.deepcopy(model).cpu().eval()
    float_model_path = MODEL_DIR / "bert_crossencoder_float.pt"
    torch.save(float_model.state_dict(), float_model_path)
    float_size_mb = model_size_mb(float_model_path)
    float_inf_sec, float_inf_ms = benchmark_latency(float_model, test_loader, torch.device("cpu"), BENCHMARK_BATCHES)
    float_row = build_row(
        "BERT Cross-Encoder",
        float_model,
        val_loader,
        test_loader,
        torch.device("cpu"),
        best_threshold,
        float_size_mb,
        int(sum(p.numel() for p in float_model.parameters())),
        float_inf_sec,
        float_inf_ms,
    )

    quant_model = torch.quantization.quantize_dynamic(
        replace_bert_embeddings_with_int8(float_model),
        {nn.Linear},
        dtype=torch.qint8,
    )
    quant_model_path = MODEL_DIR / "bert_crossencoder_int8.pt"
    torch.save(quant_model.state_dict(), quant_model_path)
    quant_size_mb = model_size_mb(quant_model_path)
    quant_inf_sec, quant_inf_ms = benchmark_latency(quant_model, test_loader, torch.device("cpu"), BENCHMARK_BATCHES)
    quant_row = build_row(
        "BERT Cross-Encoder Int8",
        quant_model,
        val_loader,
        test_loader,
        torch.device("cpu"),
        best_threshold,
        quant_size_mb,
        int(sum(p.numel() for p in quant_model.parameters())),
        quant_inf_sec,
        quant_inf_ms,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([float_row, quant_row]).to_csv(RESULT_DIR / "bert_crossencoder_quantized_results.csv", index=False)
    threshold_df.to_csv(BERT_DIR / "bert_threshold_search.csv", index=False)

    print("Saved:")
    print("-", RESULT_DIR / "bert_crossencoder_quantized_results.csv")
    print("-", BERT_DIR / "bert_threshold_search.csv")
    print(pd.DataFrame([float_row, quant_row]).to_string(index=False))


if __name__ == "__main__":
    main()
