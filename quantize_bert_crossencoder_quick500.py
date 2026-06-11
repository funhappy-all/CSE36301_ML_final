from __future__ import annotations

from quantize_bert_crossencoder_quick import (
    BATCH_SIZE,
    BENCHMARK_BATCHES,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataLoader,
    Dataset,
    EVAL_SAMPLE_SIZE,
    F,
    Int8Embedding,
    MAX_LENGTH,
    MODEL_DIR,
    OUTPUT_DIR,
    PairDataset,
    Path,
    ROOT,
    RESULT_DIR,
    accuracy_score,
    best_threshold,
    benchmark_latency,
    copy,
    f1_score,
    find_split_dir,
    load_split,
    log_loss,
    model_size_mb,
    np,
    pd,
    replace_bert_embeddings_with_int8,
    run_eval,
    torch,
)


def main() -> None:
    split_dir = find_split_dir()
    val_df = load_split(split_dir / "val.csv").head(500)
    test_df = load_split(split_dir / "test.csv").head(500)

    vocab_path = Path("/tmp/bert-base-uncased-vocab.txt")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=True)
    val_loader = DataLoader(PairDataset(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(PairDataset(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)

    checkpoint_candidates = [MODEL_DIR / "bert_crossencoder.pt", ROOT / "bert_crossencoder.pt", ROOT.parent / "bert_crossencoder.pt"]
    checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError("Missing bert_crossencoder.pt")

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

    quant_model = torch.quantization.quantize_dynamic(
        replace_bert_embeddings_with_int8(float_model),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quant_model_path = MODEL_DIR / "bert_crossencoder_int8.pt"
    torch.save(quant_model.state_dict(), quant_model_path)
    test_prob, test_y = run_eval(quant_model, test_loader, torch.device("cpu"))
    pred = (test_prob >= threshold).astype(int)
    inference_sec, inference_ms = benchmark_latency(quant_model, test_loader, torch.device("cpu"), 5)

    row = {
        "Model": "BERT Cross-Encoder Int8",
        "Owner": "Yuchan",
        "Role": "Embedding tables and linear layers quantized to int8",
        "Official Threshold Type": "validation_f1_best_sampled_500_quantization",
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
        "Benchmark Scope": "sampled_500",
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "bert_crossencoder_quantized_results.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(pd.DataFrame([row]).to_string(index=False))


if __name__ == "__main__":
    main()
