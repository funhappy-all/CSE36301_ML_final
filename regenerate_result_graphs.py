from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
TABLES_DIR = OUTPUT_DIR / "sanghun" / "tables"
FINAL_REPORT_DIR = OUTPUT_DIR / "final_report"

for path in [OUTPUT_DIR, RESULTS_DIR, TABLES_DIR, FINAL_REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "axes.titlesize": 14, "axes.labelsize": 11})


def save_both(fig, filename: str, subdir: str | None = None) -> None:
    if subdir:
        out = OUTPUT_DIR / subdir / filename
    else:
        out = OUTPUT_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(ROOT / filename, bbox_inches="tight")
    plt.close(fig)


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    candidates = [
        ROOT.parent / path.name,
        ROOT.parent / "results" / path.name,
        ROOT.parent.parent / path.name,
        ROOT.parent.parent / "results" / path.name,
        ROOT.parent.parent / "traditional_ml" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise FileNotFoundError(path)


def generate_eda() -> None:
    df = load_csv(ROOT.parent / "sanghun" / "tables" / "train_split_text.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df["is_duplicate"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "coral"], rot=0)
    axes[0].set_title("Class Distribution")
    axes[0].set_xticklabels(["Non-Duplicate", "Duplicate"])

    q1_wc = df["question1"].apply(lambda x: len(str(x).split()))
    q2_wc = df["question2"].apply(lambda x: len(str(x).split()))
    axes[1].hist(q1_wc, bins=50, alpha=0.6, label="Q1", color="steelblue")
    axes[1].hist(q2_wc, bins=50, alpha=0.6, label="Q2", color="coral")
    axes[1].set_xlim(0, 60)
    axes[1].set_title("Word Count Distribution")
    axes[1].legend()

    def _word_share(r):
        q1 = set(str(r["question1"]).lower().split())
        q2 = set(str(r["question2"]).lower().split())
        return len(q1 & q2) / max(len(q1 | q2), 1)

    ws = df.apply(_word_share, axis=1)
    ws[df.is_duplicate == 0].hist(bins=40, ax=axes[2], alpha=0.6, label="Non-Dup", color="steelblue")
    ws[df.is_duplicate == 1].hist(bins=40, ax=axes[2], alpha=0.6, label="Duplicate", color="coral")
    axes[2].set_title("Word Share Ratio")
    axes[2].legend()
    fig.tight_layout()
    save_both(fig, "eda.png")


def generate_traditional_ml_graphs() -> None:
    df = load_csv(RESULTS_DIR / "traditional_ml_results.csv")
    df = df.sort_values("Test F1", ascending=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(df["Model"], df["Test F1"], color="#4C78A8")
    ax.set_xlabel("Test F1")
    ax.set_title("Traditional ML Comparison")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(df["Test F1"]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    save_both(fig, "traditional_ml_comparison.png", subdir="results")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.barh(df["Model"], df["Test F1"], color="#B279A2")
    ax.set_xlabel("Test F1")
    ax.set_title("Traditional ML Baselines")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(df["Test F1"]):
        ax.text(value, i, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    save_both(fig, "traditional_ml_f1_comparison.png")

    feat = load_csv(ROOT.parent / "traditional_ml" / "random_forest_feature_importance.csv").sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat, x="importance", y="feature", color="steelblue", ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    fig.tight_layout()
    save_both(fig, "feature_importance.png", subdir="results")


def load_benchmark() -> pd.DataFrame:
    return load_csv(OUTPUT_DIR / "benchmark_results.csv")


def selected_models(df: pd.DataFrame) -> pd.DataFrame:
    names = [
        "Linear SVM (TF-IDF)",
        "Word2Vec + Siamese LSTM",
        "Bi-Encoder Early Stopping",
        "Embedding+Linear Int8 Bi-Encoder Early Stopping",
        "BERT Cross-Encoder",
        "BERT Cross-Encoder Int8",
    ]
    out = df[df["Model"].isin(names)].copy()
    out["Model"] = pd.Categorical(out["Model"], categories=names, ordered=True)
    return out.sort_values("Model")


def generate_tradeoff_graphs() -> None:
    df = selected_models(load_benchmark())
    labels = {
        "Linear SVM (TF-IDF)": "Best Traditional\nML",
        "Word2Vec + Siamese LSTM": "Siamese\nLSTM",
        "Bi-Encoder Early Stopping": "Bi-Encoder\n(float)",
        "Embedding+Linear Int8 Bi-Encoder Early Stopping": "Bi-Encoder\n(int8)",
        "BERT Cross-Encoder": "BERT\nCross-Encoder",
        "BERT Cross-Encoder Int8": "BERT\nCross-Enc int8",
    }
    df["Display"] = df["Model"].astype(str).map(labels)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for _, row in df.iterrows():
        ax.scatter(row["Inference ms/sample"], row["Test Accuracy"], s=120, edgecolor="white", linewidth=1.2)
        ax.annotate(row["Display"].replace("\n", " "), (row["Inference ms/sample"], row["Test Accuracy"]), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms/sample, log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Latency")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    save_both(fig, "accuracy_latency_tradeoff.png")

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for _, row in df.iterrows():
        ax.scatter(row["Model Size MB"], row["Test Accuracy"], s=120, edgecolor="white", linewidth=1.2)
        ax.annotate(row["Display"].replace("\n", " "), (row["Model Size MB"], row["Test Accuracy"]), xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Model size (MB, log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Model Size")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    save_both(fig, "accuracy_model_size_tradeoff.png")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    df2 = df.sort_values("Model Size MB", ascending=True)
    ax.barh(df2["Display"], df2["Model Size MB"], color=["#F58518" if "Bi-Encoder" in x else "#4C78A8" if "BERT" in x else "#54A24B" if "LSTM" in x else "#9E77ED" for x in df2["Display"]])
    ax.set_xlabel("Model size (MB)")
    ax.set_title("Model Size Comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_both(fig, "model_size_comparison.png")


def generate_biencoder_graphs() -> None:
    df = load_benchmark()
    df = df[df["Owner"] == "Sanghun Han"].copy()
    if df.empty:
        return
    order = [
        "Bi-Encoder",
        "Linear-Only Quantized Bi-Encoder",
        "Embedding+Linear Int8 Bi-Encoder",
        "Bi-Encoder 6 Epochs",
        "Embedding+Linear Int8 Bi-Encoder 6 Epochs",
        "Bi-Encoder Early Stopping",
        "Embedding+Linear Int8 Bi-Encoder Early Stopping",
    ]
    df["Model"] = pd.Categorical(df["Model"], categories=order, ordered=True)
    df = df.sort_values("Model")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    x = np.arange(len(df))
    ax.bar(x, df["Test F1"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Bi-Encoder", "Bi-Enc").replace("Embedding+Linear Int8 ", "Int8 ") for m in df["Model"].astype(str)], rotation=25, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_title("Bi-Encoder Compression Effect")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_both(fig, "biencoder_compression_effect.png")

    hist_path = TABLES_DIR / "biencoder_early_stopping_training_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        best_epoch = int(hist.loc[hist["val_f1"].idxmax(), "epoch"])
        best_val_f1 = float(hist["val_f1"].max())
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(hist["epoch"], hist["val_f1"], color="#4C78A8", linewidth=2.0, label="Validation F1")
        ax.plot(hist["epoch"], hist["val_accuracy"], color="#72B7B2", linewidth=1.8, label="Validation accuracy")
        ax.axvline(best_epoch, color="#E45756", linestyle="--", linewidth=1.5, label=f"Best epoch {best_epoch}")
        ax.scatter([best_epoch], [best_val_f1], color="#E45756", s=70, zorder=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation metric")
        ax.set_title("Early Stopping Curve")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_both(fig, "early_stopping_validation_curve.png")


def copy_final_report_graphs() -> None:
    mapping = [
        "all_models_f1_ranking.png",
        "final_accuracy_comparison.png",
        "final_f1_comparison.png",
        "final_f1_vs_latency.png",
        "final_f1_vs_logloss.png",
        "final_f1_vs_size.png",
        "final_inference_speed_comparison.png",
        "final_logloss_comparison.png",
        "final_model_size_comparison.png",
        "final_parameter_count_comparison.png",
    ]
    for name in mapping:
        src = FINAL_REPORT_DIR / name
        if src.exists():
            shutil.copy2(src, ROOT / name)
            shutil.copy2(src, OUTPUT_DIR / name)


def main() -> None:
    generate_eda()
    generate_traditional_ml_graphs()
    generate_tradeoff_graphs()
    generate_biencoder_graphs()
    copy_final_report_graphs()
    print("Regenerated graphs.")


if __name__ == "__main__":
    main()
