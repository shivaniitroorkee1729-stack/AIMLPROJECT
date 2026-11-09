# Class Confusion Analysis for Historically Problematic Pairs

Detailed directional error rates and mutual confusion on the held-out facial crops test split.

### Pair: Hasya vs. Bhayanaka

- **True Hasya Samples:** 248 | **True Bhayanaka Samples:** 242

| Model | True Hasya -> Pred Bhayanaka | True Bhayanaka -> Pred Hasya | Total Mutual Errors | Mutual Error Rate |
|:---|:---:|:---:|:---:|:---:|
| Model A (CLIP + LogReg) | 0 (0.0%) | 0 (0.0%) | 0 | **0.0%** |
| Model B (CLIP + SVM) | 0 (0.0%) | 0 (0.0%) | 0 | **0.0%** |
| Model C (EfficientNet-B0) | 2 (0.81%) | 0 (0.0%) | 2 | **0.41%** |

### Pair: Karuna vs. Shanta

- **True Karuna Samples:** 247 | **True Shanta Samples:** 244

| Model | True Karuna -> Pred Shanta | True Shanta -> Pred Karuna | Total Mutual Errors | Mutual Error Rate |
|:---|:---:|:---:|:---:|:---:|
| Model A (CLIP + LogReg) | 0 (0.0%) | 0 (0.0%) | 0 | **0.0%** |
| Model B (CLIP + SVM) | 0 (0.0%) | 0 (0.0%) | 0 | **0.0%** |
| Model C (EfficientNet-B0) | 1 (0.4%) | 0 (0.0%) | 1 | **0.2%** |

### Pair: Raudra vs. Veera

- **True Raudra Samples:** 238 | **True Veera Samples:** 232

| Model | True Raudra -> Pred Veera | True Veera -> Pred Raudra | Total Mutual Errors | Mutual Error Rate |
|:---|:---:|:---:|:---:|:---:|
| Model A (CLIP + LogReg) | 4 (1.68%) | 4 (1.72%) | 8 | **1.7%** |
| Model B (CLIP + SVM) | 2 (0.84%) | 2 (0.86%) | 4 | **0.85%** |
| Model C (EfficientNet-B0) | 3 (1.26%) | 6 (2.59%) | 9 | **1.91%** |

### Pair: Adbhuta vs. Bhayanaka

- **True Adbhuta Samples:** 223 | **True Bhayanaka Samples:** 242

| Model | True Adbhuta -> Pred Bhayanaka | True Bhayanaka -> Pred Adbhuta | Total Mutual Errors | Mutual Error Rate |
|:---|:---:|:---:|:---:|:---:|
| Model A (CLIP + LogReg) | 2 (0.9%) | 4 (1.65%) | 6 | **1.29%** |
| Model B (CLIP + SVM) | 0 (0.0%) | 0 (0.0%) | 0 | **0.0%** |
| Model C (EfficientNet-B0) | 1 (0.45%) | 5 (2.07%) | 6 | **1.29%** |

