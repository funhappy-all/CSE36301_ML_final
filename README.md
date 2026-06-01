# CSE36301 ML Final Package

This folder is the cleaned, reproducible package for the final project.

It contains only the files needed to rerun the experiments and rebuild the final comparison figures.

## Contents

- `notebooks/`
  - `01_preprocessing.ipynb`
  - `02_traditional_ml.ipynb`
  - `03_siamese_lstm_jun_data_only.ipynb`
  - `04_biencoder_quantization_sanghun.ipynb`
- `generate_final_benchmark_report.py`
- `final_model_comparison.md`
- `final_model_comparison.csv`
- final comparison PNG figures

## How to reproduce

1. Place the Quora Question Pairs data at the repository root.
   - Either `data/train.csv` or the original Kaggle archive under `quora-question-pairs/` is fine.
2. Run `notebooks/01_preprocessing.ipynb`.
   - Creates the shared split files and preprocessing outputs.
3. Run `notebooks/02_traditional_ml.ipynb`.
   - Builds the Traditional ML baseline results and plots.
4. Run `notebooks/03_siamese_lstm_jun_data_only.ipynb`.
   - Builds the Siamese LSTM result files.
   - This notebook intentionally does not generate the final comparison plots.
5. Run `notebooks/04_biencoder_quantization_sanghun.ipynb`.
   - Builds the Bi-Encoder, quantization, and benchmark outputs.
6. Run `python generate_final_benchmark_report.py`.
   - Rebuilds the final comparison table and final report figures in `outputs/final_report/`.

## Final report files

The report-ready outputs are:

- `final_model_comparison.md`
- `final_model_comparison.csv`
- `final_accuracy_comparison.png`
- `final_f1_comparison.png`
- `final_logloss_comparison.png`
- `final_parameter_count_comparison.png`
- `final_inference_speed_comparison.png`
- `final_model_size_comparison.png`
- `final_f1_vs_size.png`
- `final_f1_vs_latency.png`
- `final_f1_vs_logloss.png`
- `all_models_f1_ranking.png`

## Notes

- `outputs/final_report/` is the full source of truth for the benchmark results.
- `outputs/CSE36301_ML_final/` is the curated package intended for GitHub upload and reruns.
- The best Traditional ML baseline is `Linear SVM (TF-IDF)`.
- The main Sanghun result is the early-stopped Bi-Encoder.
- The int8 Bi-Encoder is the compressed deployment variant.
