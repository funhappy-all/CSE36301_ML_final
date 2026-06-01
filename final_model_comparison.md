# Final Model Comparison

This table is the cleaned final shortlist for the report.

| Model | Owner | Accuracy | F1 | Log Loss | Parameters | Inference ms/sample | Model Size MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear SVM (TF-IDF) | Taehun Kim | 0.7455 | 0.6395 | 0.5918 | 93,013 | 0.000026 | 0.710 |
| Word2Vec + Siamese LSTM | Jun Park | 0.7759 | 0.7238 | 0.4646 | 4,565,357 | 0.281105 | 52.308 |
| Bi-Encoder Early Stopping | Sanghun Han | 0.8217 | 0.7747 | 0.5109 | 1,304,897 | 0.012353 | 4.982 |
| Embedding+Linear Int8 Bi-Encoder Early Stopping | Sanghun Han | 0.8163 | 0.7723 | 0.5193 | 1,304,897 | 0.014884 | 1.254 |

## Final takeaway
- Best overall F1: **Bi-Encoder Early Stopping** (`0.7747`), which is also the strongest accuracy-quality tradeoff.
- Smallest shortlisted model: **Linear SVM (TF-IDF)** (`0.710 MB`).
- Fastest shortlisted model: **Linear SVM (TF-IDF)** (`0.000026 ms/sample`).
- The int8 Bi-Encoder keeps almost the same F1 as the float Bi-Encoder while cutting model size sharply, so it is the best deployment candidate.