from __future__ import annotations

from quantize_bert_crossencoder_quick import (
    BATCH_SIZE,
    BENCHMARK_BATCHES,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataLoader,
    PairDataset,
    Path,
    ROOT,
    RESULT_DIR,
    accuracy_score,
    best_threshold,
    benchmark_latency,
    load_split,
    log_loss,
    model_size_mb,
    np,
    pd,
    replace_bert_embeddings_with_int8,
    run_eval,
    torch,
    f1_score,
)


def main() -> None:
    split_dir = next((d for d in [ROOT / "for_bert", OUTPUT_DIR / "for_bert", ROOT.parent / "for_bert", ROOT.parent.parent / "for_bert"] if (d / "train.csv").exists()), None)
    if split_dir is None:
        raise FileNotFoundError("Could not find for_bert split CSVs.")

    train_df = load_split(split_dir / "train.csv")
    val_df = load_split(split_dir / "val.csv").head(500)
    test_df = load_split(split_dir / "test.csv").head(500)

    vocab_path = Path("/tmp/bert-base-uncased-vocab.txt")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    val_loader = DataLoader(PairDataset(val_df, tokenizer, 128), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairDataset(test_df, tokenizer, 128), batch_size=BATCH_SIZE, shuffle=False)

    checkpoint_candidates = [OUTPUT_DIR / "models" / "bert_crossencoder_float.pt", OUTPUT_DIR / "models" / "bert_crossencoder.pt", ROOT / "bert_crossencoder.pt", ROOT.parent / "bert_crossencoder.pt"]
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

    float_size_path = OUTPUT_DIR / "models" / "bert_crossencoder_float_sampled500_fair.pt"
    torch.save(float_model.state_dict(), float_size_path)

    int8_model = torch.quantization.quantize_dynamic(replace_bert_embeddings_with_int8(float_model), {torch.nn.Linear}, dtype=torch.qint8)
    int8_size_path = OUTPUT_DIR / "models" / "bert_crossencoder_int8_sampled500_fair.pt"
    torch.save(int8_model.state_dict(), int8_size_path)

    def make_row(model_name: str, model: torch.nn.Module, size_path: Path) -> dict[str, object]:
        test_prob, test_y = run_eval(model, test_loader, torch.device("cpu"))
        infer_sec, infer_ms = benchmark_latency(model, test_loader, torch.device("cpu"), BENCHMARK_BATCHES)
        pred = (test_prob >= threshold).astype(int)
        return {
            "Model": model_name,
            "Owner": "Yuchan",
            "Role": "Embedding tables and linear layers quantized to int8" if "Int8" in model_name else "Float BERT Cross-Encoder",
            "Official Threshold Type": "validation_f1_best_sampled_500_fair",
            "Best Threshold": threshold,
            "Accuracy Threshold": threshold,
            "Train Samples": int(len(train_df)),
            "Validation Samples": int(len(val_df)),
            "Test Samples": int(len(test_df)),
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
            "Max Sequence Length": 128,
            "Vocabulary Size": 30522,
            "Embedding Dim": 768,
            "LSTM Units": None,
            "Batch Size": BATCH_SIZE,
            "Epochs Run": 3,
            "GPU": "CPU only",
            "Benchmark Scope": "sampled_500",
        }

    rows = [
        make_row("BERT Cross-Encoder", float_model, float_size_path),
        make_row("BERT Cross-Encoder Int8", int8_model, int8_size_path),
    ]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "bert_crossencoder_quantized_results_fair.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows)[["Model", "Benchmark Scope", "Validation Samples", "Test Samples", "Test Accuracy", "Test F1", "Inference ms/sample", "Model Size MB"]].to_string(index=False))
    print("Saved:", out_path)


if __name__ == "__main__":
    from quantize_bert_crossencoder_quick import OUTPUT_DIR
    main()
