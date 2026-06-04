# Notebook Package

This folder contains the notebook set required to reproduce the final benchmark.

## Included notebooks

- `01_preprocessing.ipynb`
  - shared preprocessing and split generation
- `02_traditional_ml.ipynb`
  - Traditional ML baselines and plots
- `03_siamese_lstm_jun_data_only.ipynb`
  - Siamese LSTM result generation
- `04_biencoder_quantization_sanghun.ipynb`
  - Bi-Encoder, quantization, and final benchmarking
- `05_bert_crossencoder.ipynb`
  - BERT Cross-Encoder baseline

## Excluded notebooks

- `03_siamese_lstm.ipynb`
  - legacy duplicate of the Jun baseline
- `04_bert_crossencoder.ipynb`
  - not used in the final report
- `05_biencoder_quantization.ipynb`
  - legacy duplicate of the Sanghun notebook

## Execution notes

- Run the notebooks in the order listed above.
- The packaged notebooks are saved without execution outputs.
- The final comparison charts are rebuilt by `../generate_final_benchmark_report.py`.
- `03_siamese_lstm_jun_data_only.ipynb` is intentionally data-only and does not create the final comparison figures.
- `05_bert_crossencoder.ipynb` includes the BERT baseline code used for the final comparison.
