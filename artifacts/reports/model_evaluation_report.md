# Navarasa AI: Comprehensive Model Evaluation & Benchmark Report

**Test Set Split:** Held-Out Facial Crops Test Set (2109 samples)

**Selected Primary Model for Deployment:** **Model B (CLIP + SVM)**

> Model B (CLIP + SVM) achieved the highest test accuracy (99.19%) and highest macro F1-score (99.19%) on held-out facial crops, virtually resolving confusion between Hasya and Bhayanaka.

## 1. Overall Performance Benchmark

| Evaluation Metric | Model A: CLIP + LogReg | Model B: CLIP + SVM (RBF) | Model C: EfficientNet-B0 |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | **97.44%** | **99.19%** | **95.02%** |
| **Macro Precision** | 97.46% | 99.19% | 95.01% |
| **Macro Recall** | 97.44% | 99.18% | 95.06% |
| **Macro F1-Score** | 97.43% | 99.19% | 95.02% |
| **Weighted F1-Score** | 97.44% | 99.19% | 95.02% |
| **Inference Latency** | 0.01 ms | 6.40 ms | 58.89 ms |

## 2. Per-Class F1-Score Breakdown (All 9 Navarasas)

| Navarasa Class | Support | Model A (LogReg) F1 | Model B (SVM) F1 | Model C (EffNet) F1 |
|:---|:---:|:---:|:---:|:---:|
| **Shringara** | 202 | 97.77% | 98.76% | 95.10% |
| **Hasya** | 248 | 98.18% | 99.19% | 93.06% |
| **Karuna** | 247 | 98.37% | 99.60% | 96.52% |
| **Raudra** | 238 | 96.49% | 98.95% | 94.96% |
| **Veera** | 232 | 96.41% | 98.29% | 92.99% |
| **Bhayanaka** | 242 | 95.78% | 99.17% | 91.77% |
| **Bibhatsa** | 233 | 96.55% | 98.92% | 94.32% |
| **Adbhuta** | 223 | 97.75% | 100.00% | 96.90% |
| **Shanta** | 244 | 99.59% | 99.79% | 99.59% |
