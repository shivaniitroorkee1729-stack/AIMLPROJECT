# Navarasa Facial Expression Dataset Analysis Report

- **Total Valid Images:** 14805
- **Total Classes:** 9
- **Corrupted Images:** 0
- **Duplicate Image Sets:** 473 (705 redundant files)

## 1. Class Distribution

| Index | Navarasa Class | Image Count | Percentage (%) |
|-------|----------------|-------------|----------------|
| 1 | **Shringara** | 1605 | 10.84% |
| 2 | **Hasya** | 1698 | 11.47% |
| 3 | **Karuna** | 1650 | 11.14% |
| 4 | **Raudra** | 1621 | 10.95% |
| 5 | **Veera** | 1690 | 11.42% |
| 6 | **Bhayanaka** | 1630 | 11.01% |
| 7 | **Bibhatsa** | 1610 | 10.87% |
| 8 | **Adbhuta** | 1650 | 11.14% |
| 9 | **Shanta** | 1651 | 11.15% |

## 2. Image Dimensions & Formats

- **Formats:** {'JPEG': 14805}
- **Resolution Range:** 202x176 to 1920x1080
- **Median Resolution:** 470x506
- **Mean Aspect Ratio:** 1.02

## 3. Data Integrity & Quality Observations

- **Zero unreadable/corrupted files:** All extracted images passed PIL integrity validation.
- **Duplicate Analysis:** 473 sets of identical files detected. The stratified dataset split handles duplicates to prevent cross-split leakage.
