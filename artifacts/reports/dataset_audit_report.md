# Navarasa Facial Expression Dataset Audit Report

## 1. Executive Summary
- **Total Image Files Audited:** 14,805
- **Unique Disjoint Samples:** 14,100
- **Exact Duplicate Files Detected & Filtered:** 705
- **Corrupted / Unreadable Images:** 0
- **Classes:** 9 Canonical Navarasas

## 2. Per-Class Image Count & Quality Overview

| Navarasa Class | Total Files | Unique Samples | Duplicates | Avg Resolution | Single-Face Detection Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Shringara** | 1,605 | 1,345 | 260 | 610×555 px | 100.0% |
| **Hasya** | 1,698 | 1,664 | 34 | 469×498 px | 100.0% |
| **Karuna** | 1,650 | 1,646 | 4 | 494×494 px | 100.0% |
| **Raudra** | 1,621 | 1,595 | 26 | 462×496 px | 100.0% |
| **Veera** | 1,690 | 1,566 | 124 | 862×665 px | 100.0% |
| **Bhayanaka** | 1,630 | 1,610 | 20 | 540×521 px | 100.0% |
| **Bibhatsa** | 1,610 | 1,559 | 51 | 665×563 px | 100.0% |
| **Adbhuta** | 1,650 | 1,486 | 164 | 536×528 px | 100.0% |
| **Shanta** | 1,651 | 1,629 | 22 | 727×591 px | 98.67% |

## 3. Confusing Class Pairs Diagnostic Analysis

### Hasya vs Bhayanaka
- **Anatomical & FACS Distinction:** Hasya displays AU6 (cheek raiser) + AU12 (lip corner puller) with crinkled eyes; Bhayanaka displays AU5 (upper lid raiser) + AU1+2 (brow raiser) with tense open mouth without cheek elevation.
- **Sample Counts:** Hasya (1,698) vs Bhayanaka (1,630)

### Karuna vs Shanta
- **Anatomical & FACS Distinction:** Karuna features inner brow elevation (AU1) and downturned lip corners (AU15) denoting sorrow; Shanta features smooth brows, balanced relaxed lips, and neutral calmness.
- **Sample Counts:** Karuna (1,650) vs Shanta (1,651)

### Raudra vs Veera
- **Anatomical & FACS Distinction:** Raudra features severe brow furrowing (AU4), intense glaring eyes, and anger tension; Veera exhibits poised elevated eyebrows (AU1+2), firm jaw, confident dignified tension without wrath.
- **Sample Counts:** Raudra (1,621) vs Veera (1,690)

### Adbhuta vs Bhayanaka
- **Anatomical & FACS Distinction:** Adbhuta exhibits circular surprise with raised arched brows and relaxed dropped jaw (AU26); Bhayanaka displays wide terror with horizontal mouth tension, lip stretch (AU20), and sclera exposure.
- **Sample Counts:** Adbhuta (1,650) vs Bhayanaka (1,630)

## 4. Key Takeaways & Preprocessing Directives for Retraining
1. **Face Crop Domain Alignment:** Previous models trained on uncropped raw images suffered domain mismatch when presented with tight facial crops on the website. All retraining must directly operate on standardized facial crops.
2. **Facial Geometry Filtering:** Images with 0 detectable faces or multi-subject ambiguity should be pruned from the training splits to prevent noisy gradient updates.
3. **Balanced Stratification:** Class proportions are well-balanced across all 9 classes (ranging between 1,605 and 1,698 samples each).
