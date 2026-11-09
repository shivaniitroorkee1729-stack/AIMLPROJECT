# Navarasa Models Evaluation & Comparison Report

Evaluation conducted on the strictly held-out test split of **2,115 unseen images**.

## Overall Performance Summary

| Metric | CLIP ViT-B/32 + LogReg | EfficientNet-B0 (Fine-Tuned) | Difference |
|--------|-------------------------|------------------------------|------------|
| **Test Accuracy** | 97.40% | 94.99% | -2.41% |
| **Macro Precision** | 97.37% | 94.99% | -2.38% |
| **Macro Recall** | 97.39% | 95.04% | -2.34% |
| **Macro F1-Score** | 97.37% | 95.00% | -2.37% |
| **Weighted F1-Score** | 97.40% | 94.97% | -2.43% |

## Per-Class F1-Score Breakdown

| Navarasa Class | CLIP Precision | CLIP Recall | CLIP F1 | EffNet Precision | EffNet Recall | EffNet F1 |
|----------------|----------------|-------------|---------|------------------|---------------|-----------|
| **Shringara** | 95.6% | 97.0% | 96.3% | 95.1% | 97.0% | 96.1% |
| **Hasya** | 97.9% | 95.6% | 96.7% | 92.1% | 88.4% | 90.2% |
| **Karuna** | 98.4% | 99.6% | 99.0% | 95.2% | 95.5% | 95.4% |
| **Raudra** | 94.7% | 97.5% | 96.1% | 94.8% | 99.2% | 96.9% |
| **Veera** | 97.9% | 97.0% | 97.4% | 95.7% | 95.7% | 95.7% |
| **Bhayanaka** | 97.1% | 96.7% | 96.9% | 94.3% | 95.5% | 94.9% |
| **Bibhatsa** | 98.2% | 94.9% | 96.5% | 93.8% | 90.6% | 92.2% |
| **Adbhuta** | 96.9% | 98.2% | 97.6% | 96.4% | 96.0% | 96.2% |
| **Shanta** | 99.6% | 100.0% | 99.8% | 97.5% | 97.5% | 97.5% |
