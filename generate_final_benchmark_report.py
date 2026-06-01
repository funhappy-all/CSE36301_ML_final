from __future__ import annotations

import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "final_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(REPORT_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def load_benchmark_table() -> pd.DataFrame:
    path = OUTPUT_DIR / "benchmark_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark table: {path}")

    df = pd.read_csv(path)
    required = [
        "Model",
        "Owner",
        "Test Accuracy",
        "Test F1",
        "Test Log Loss",
        "Inference ms/sample",
        "Model Size MB",
        "Parameter Count",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Benchmark table is missing columns: {missing}")
    return df


def fmt_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{int(round(float(value))):,}"


def pick_selected_models(df: pd.DataFrame) -> pd.DataFrame:
    chosen = []

    traditional = df[df["Owner"] == "Taehun Kim"].sort_values("Test F1", ascending=False).head(1)
    if len(traditional):
        chosen.append(traditional.iloc[0])

    lstm = df[df["Model"] == "Word2Vec + Siamese LSTM"]
    if len(lstm):
        chosen.append(lstm.iloc[0])

    float_bi = df[df["Model"] == "Bi-Encoder Early Stopping"]
    if len(float_bi):
        chosen.append(float_bi.iloc[0])
    else:
        float_bi = (
            df[df["Owner"] == "Sanghun Han"]
            .query("`Model`.str.contains('Bi-Encoder', case=False, regex=False)", engine="python")
            .sort_values("Test F1", ascending=False)
            .head(1)
        )
        if len(float_bi):
            chosen.append(float_bi.iloc[0])

    int8_bi = df[df["Model"] == "Embedding+Linear Int8 Bi-Encoder Early Stopping"]
    if len(int8_bi):
        chosen.append(int8_bi.iloc[0])
    else:
        int8_bi = (
            df[df["Owner"] == "Sanghun Han"]
            .query("`Model`.str.contains('Int8', case=False, regex=False)", engine="python")
            .sort_values("Test F1", ascending=False)
            .head(1)
        )
        if len(int8_bi):
            chosen.append(int8_bi.iloc[0])

    selected = pd.DataFrame(chosen).copy()
    order = [
        "Linear SVM (TF-IDF)",
        "Word2Vec + Siamese LSTM",
        "Bi-Encoder Early Stopping",
        "Embedding+Linear Int8 Bi-Encoder Early Stopping",
    ]
    selected["sort_key"] = pd.Categorical(selected["Model"], categories=order, ordered=True)
    selected = selected.sort_values("sort_key").drop(columns=["sort_key"])
    return selected


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Model Display"] = out["Model"].replace(
        {
            "Linear SVM (TF-IDF)": "Best Traditional\nML",
            "Word2Vec + Siamese LSTM": "Siamese\nLSTM",
            "Bi-Encoder Early Stopping": "Bi-Encoder\n(float)",
            "Embedding+Linear Int8 Bi-Encoder Early Stopping": "Bi-Encoder\n(int8)",
        }
    )
    return out


def add_display_parameter_count(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Parameter Count (Display)"] = out["Parameter Count"].astype(float)

    float_reference = None
    float_model = out[out["Model"] == "Bi-Encoder Early Stopping"]
    if len(float_model):
        float_reference = float(float_model.iloc[0]["Parameter Count"])

    for idx, row in out.iterrows():
        param_count = row.get("Parameter Count", np.nan)
        if pd.notna(param_count) and float(param_count) > 0:
            out.at[idx, "Parameter Count (Display)"] = float(param_count)
            continue

        if row.get("Model") == "Embedding+Linear Int8 Bi-Encoder Early Stopping" and float_reference is not None:
            out.at[idx, "Parameter Count (Display)"] = float_reference
            continue

        state_count = row.get("State Tensor Count", np.nan)
        if pd.notna(state_count) and float(state_count) > 0:
            out.at[idx, "Parameter Count (Display)"] = float(state_count)
            continue

        encoder_count = row.get("Encoder Parameter Count", np.nan)
        if pd.notna(encoder_count) and float(encoder_count) > 0:
            out.at[idx, "Parameter Count (Display)"] = float(encoder_count)

    return out


def save_barplot(
    df: pd.DataFrame,
    metric: str,
    filename: str,
    title: str,
    ylabel: str,
    value_fmt: str,
    xscale: str | None = None,
    highlight_best: bool = True,
) -> None:
    plot_df = df.sort_values(metric, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    palette = ["#9E77ED" if row["Owner"] == "Sanghun Han" else "#4C78A8" if row["Owner"] == "Jun Park" else "#F58518" for _, row in plot_df.iterrows()]
    bars = ax.barh(plot_df["Model Display"], plot_df[metric].astype(float), color=palette, edgecolor="white", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("")
    if xscale:
        ax.set_xscale(xscale)

    if highlight_best and len(plot_df):
        best_idx = plot_df[metric].astype(float).idxmax()
        best_pos = list(plot_df.index).index(best_idx)
        bars[best_pos].set_color("#54A24B")

    for bar, value in zip(bars, plot_df[metric].astype(float)):
        ax.text(
            bar.get_width() * (1.01 if (not xscale or xscale == "linear") else 1.15),
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(value),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xscale: str | None = None,
    yscale: str | None = None,
    size_col: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    colors = {
        "Taehun Kim": "#F58518",
        "Jun Park": "#4C78A8",
        "Sanghun Han": "#9E77ED",
    }
    sizes = None
    if size_col:
        raw = df[size_col].astype(float).to_numpy()
        scaled = np.interp(raw, (raw.min(), raw.max()), (120, 700))
        sizes = scaled

    for idx, row in df.iterrows():
        ax.scatter(
            row[x],
            row[y],
            s=sizes[list(df.index).index(idx)] if sizes is not None else 160,
            color=colors.get(row["Owner"], "#72B7B2"),
            edgecolor="white",
            linewidth=1.2,
            alpha=0.92,
        )
        ax.annotate(
            row["Model Display"].replace("\n", " "),
            (row[x], row[y]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )

    if xscale:
        ax.set_xscale(xscale)
    if yscale:
        ax.set_yscale(yscale)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def build_summary_tables(selected: pd.DataFrame, all_models: pd.DataFrame) -> None:
    selected_out = selected.copy()
    selected_out["Parameter Count (Raw)"] = selected_out["Parameter Count"].round().astype("Int64")
    selected_out["Parameter Count"] = selected_out["Parameter Count (Display)"].round().astype("Int64")
    selected_out["Test Log Loss"] = selected_out["Test Log Loss"].astype(float)
    selected_out["Inference ms/sample"] = selected_out["Inference ms/sample"].astype(float)
    selected_out["Model Size MB"] = selected_out["Model Size MB"].astype(float)

    all_models_out = all_models.copy()
    all_models_out["Parameter Count (Raw)"] = all_models_out["Parameter Count"].round().astype("Int64")
    all_models_out["Parameter Count"] = all_models_out["Parameter Count (Display)"].round().astype("Int64")

    cols = [
        "Model",
        "Owner",
        "Test Accuracy",
        "Test F1",
        "Test Log Loss",
        "Parameter Count",
        "Parameter Count (Raw)",
        "Inference ms/sample",
        "Model Size MB",
    ]
    selected_out[cols].to_csv(REPORT_DIR / "final_model_comparison.csv", index=False)
    all_models_out.sort_values("Test F1", ascending=False)[cols].to_csv(
        REPORT_DIR / "all_models_sorted_by_f1.csv", index=False
    )

    md_lines = []
    md_lines.append("# Final Model Comparison")
    md_lines.append("")
    md_lines.append("This table is the cleaned final shortlist for the report.")
    md_lines.append("")
    md_lines.append("| Model | Owner | Accuracy | F1 | Log Loss | Parameters | Inference ms/sample | Model Size MB |")
    md_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in selected_out.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['Owner']} | {row['Test Accuracy']:.4f} | {row['Test F1']:.4f} | {row['Test Log Loss']:.4f} | {fmt_int(row['Parameter Count'])} | {row['Inference ms/sample']:.6f} | {row['Model Size MB']:.3f} |"
        )

    best = selected_out.sort_values("Test F1", ascending=False).iloc[0]
    small = selected_out.sort_values("Model Size MB", ascending=True).iloc[0]
    fast = selected_out.sort_values("Inference ms/sample", ascending=True).iloc[0]
    md_lines.append("")
    md_lines.append("## Final takeaway")
    md_lines.append(
        f"- Best overall F1: **{best['Model']}** (`{best['Test F1']:.4f}`), which is also the strongest accuracy-quality tradeoff."
    )
    md_lines.append(
        f"- Smallest shortlisted model: **{small['Model']}** (`{small['Model Size MB']:.3f} MB`)."
    )
    md_lines.append(
        f"- Fastest shortlisted model: **{fast['Model']}** (`{fast['Inference ms/sample']:.6f} ms/sample`)."
    )
    md_lines.append(
        "- The int8 Bi-Encoder keeps almost the same F1 as the float Bi-Encoder while cutting model size sharply, so it is the best deployment candidate."
    )
    (REPORT_DIR / "final_model_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    df = load_benchmark_table()
    selected = add_display_columns(pick_selected_models(df))
    selected = add_display_parameter_count(selected)
    all_models = add_display_columns(df.copy())
    all_models = add_display_parameter_count(all_models)

    build_summary_tables(selected, all_models)

    plot_df = selected.copy()
    save_barplot(plot_df, "Test Accuracy", "final_accuracy_comparison.png", "Final Model Accuracy Comparison", "Accuracy", "{:.3f}")
    save_barplot(plot_df, "Test F1", "final_f1_comparison.png", "Final Model F1 Comparison", "F1", "{:.3f}")
    save_barplot(plot_df, "Test Log Loss", "final_logloss_comparison.png", "Final Model Log Loss Comparison", "Log loss", "{:.4f}", highlight_best=False)
    save_barplot(plot_df, "Parameter Count (Display)", "final_parameter_count_comparison.png", "Final Model Parameter Count Comparison", "Parameters (log scale)", "{:,}", xscale="log", highlight_best=False)
    save_barplot(plot_df, "Inference ms/sample", "final_inference_speed_comparison.png", "Final Model Inference Speed Comparison", "Inference time (ms/sample, log scale)", "{:.6f}", xscale="log", highlight_best=False)
    save_barplot(plot_df, "Model Size MB", "final_model_size_comparison.png", "Final Model Size Comparison", "Model size (MB, log scale)", "{:.3f}", xscale="log", highlight_best=False)

    save_scatter(
        plot_df,
        x="Model Size MB",
        y="Test F1",
        filename="final_f1_vs_size.png",
        title="F1 vs Model Size",
        xlabel="Model size (MB, log scale)",
        ylabel="Test F1",
        xscale="log",
        size_col="Parameter Count (Display)",
    )
    save_scatter(
        plot_df,
        x="Inference ms/sample",
        y="Test F1",
        filename="final_f1_vs_latency.png",
        title="F1 vs Inference Latency",
        xlabel="Inference latency (ms/sample, log scale)",
        ylabel="Test F1",
        xscale="log",
        size_col="Model Size MB",
    )
    save_scatter(
        plot_df,
        x="Test Log Loss",
        y="Test F1",
        filename="final_f1_vs_logloss.png",
        title="F1 vs Log Loss",
        xlabel="Test Log Loss",
        ylabel="Test F1",
        size_col="Model Size MB",
    )

    # A compact appendix view for the full benchmark table.
    appendix = df.sort_values("Test F1", ascending=False).copy()
    appendix["Model Display"] = appendix["Model"].str.replace("Bi-Encoder", "Bi-Encoder", regex=False)
    fig, ax = plt.subplots(figsize=(10.4, 6.8))
    ax.barh(appendix["Model"].iloc[::-1], appendix["Test F1"].iloc[::-1], color="#72B7B2")
    ax.set_title("All Benchmark Models by Test F1")
    ax.set_xlabel("Test F1")
    ax.set_ylabel("")
    for i, value in enumerate(appendix["Test F1"].iloc[::-1]):
        ax.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "all_models_f1_ranking.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved final report to {REPORT_DIR}")


if __name__ == "__main__":
    main()
