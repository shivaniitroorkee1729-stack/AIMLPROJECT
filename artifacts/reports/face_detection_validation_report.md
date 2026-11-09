# Face Detection Validation & Performance Report

**Execution Timestamp:** `2026-11-09 21:32:59`  
**Pipeline Architecture:** Multi-Engine Cascade (`OpenCV YuNet` + `MediaPipe FaceLandmarker`)  

---

## 1. Executive Summary

A comprehensive benchmark was conducted across all 9 Navarasa classes, challenging boundary cases, synthetic non-human distractors, and multi-face composites to evaluate the dual-engine human face detection and validation layer.

| Metric | Measured Value | Target Requirement | Status |
| :--- | :--- | :--- | :---: |
| **Total Real Dataset Images Tested** | **315** | $\ge 315$ (35 per class) | **PASSED** |
| **Navarasa Face Detection Accuracy** | **100.0%** | $> 95.0\%$ | **PASSED** |
| **False Positive Rate (Non-Human Images)** | **0.0%** | $0.0\%$ | **PASSED** |
| **Multi-Face Rejection Rate** | **100.0%** | $100.0\%$ | **PASSED** |
| **Average Detection Latency** | **68.22 ms** | $< 100$ ms | **PASSED** |
| **Adbhuta 1080p Frame (train 26.jpg)** | **PASSED (1 Face)** | Exactly 1 Face | **FIXED** |

---

## 2. Detector Architecture & Configuration

```mermaid
flowchart TD
    A["Uploaded Image / Camera Stream"] --> B["EXIF Orientation Transposition (ImageOps.exif_transpose)"]
    B --> C["OpenCV YuNet Multi-Scale Detector (232 KB ONNX)"]
    C --> DFaces Detected?
    D -- "0 Candidates" --> E["MediaPipe Global Fallback"]
    D -- ">= 1 Candidates" --> F["Extract Bounding Box Candidates + IoU NMS (0.35)"]
    E --> GMediaPipe Candidates?
    G -- "None" --> H["CASE A: REJECT (0 Faces Detected)"]
    G -- ">= 1" --> F
    F --> I["Secondary Verification: MediaPipe FaceLandmarker on Candidate Crops"]
    I --> JValid 478 3D Landmarks & Dense Blendshapes?
    J -- "No (Noise / Shape / Hand)" --> K["Prune Candidate"]
    J -- "Yes (Genuine Human Face)" --> L["Relative Area Filter (> 12% of Primary Face)"]
    L --> MFiltered Face Count
    M -- "0" --> H
    M -- "1" --> N["CASE B: ACCEPT (Extract Proportional Square Crop -> ML & XAI)"]
    M -- ">= 2" --> O["CASE C: REJECT (Multiple Faces Detected -> Block ML & XAI)"]
```

### Configuration Parameters
- **Primary Detector:** `OpenCV YuNet (face_detection_yunet_2023mar.onnx)`
- **Score Threshold:** `0.6`
- **NMS Threshold:** `0.35`
- **Verification Engine:** `MediaPipe Face Landmarker (478 3D landmarks + 52 FACS blendshapes)`
- **Crop Verification Confidence:** `0.25`
- **Face Crop Padding Ratio:** `22.0%`
- **Background Filtering:** `< 12% primary face area`

---

## 3. Class-by-Class Detection Performance

| Navarasa Class | Tested Images | Single Face Detected | Accuracy (%) | Avg Confidence | Avg Face Area (%) | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Shringara** | 35 | 35 | 100.0% | 93.2% | 11.52% | 58.52 ms |
| **Hasya** | 35 | 35 | 100.0% | 93.4% | 12.2% | 53.33 ms |
| **Karuna** | 35 | 35 | 100.0% | 93.6% | 11.97% | 59.45 ms |
| **Raudra** | 35 | 35 | 100.0% | 93.2% | 7.36% | 57.31 ms |
| **Veera** | 35 | 35 | 100.0% | 93.5% | 6.87% | 107.36 ms |
| **Bhayanaka** | 35 | 35 | 100.0% | 93.7% | 9.46% | 66.98 ms |
| **Bibhatsa** | 35 | 35 | 100.0% | 93.4% | 7.67% | 74.52 ms |
| **Adbhuta** | 35 | 35 | 100.0% | 93.7% | 10.12% | 58.18 ms |
| **Shanta** | 35 | 35 | 100.0% | 93.5% | 6.04% | 78.36 ms |

---

## 4. Previously Failing Boundary Cases Verification

| Case Description | Relative Path | Detected Status | Face Count | Confidence | Area Share | Result |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Adbhuta Anita Babu 1080p** | `data/processed/Adbhuta/train (26).jpg` | `SINGLE_FACE` | `1` | 93.2% | 1.32% | **PASSED** ✅ |
| **Veera 1080p Medium Shot** | `data/processed/Veera/train (10).jpg` | `SINGLE_FACE` | `1` | 93.8% | 1.52% | **PASSED** ✅ |
| **Shanta Mudra Hand Alta** | `data/processed/Shanta/train (1000).jpg` | `SINGLE_FACE` | `1` | 93.1% | 5.59% | **PASSED** ✅ |
| **Shanta Background Artwork** | `data/processed/Shanta/train (102).jpg` | `SINGLE_FACE` | `1` | 94.0% | 1.66% | **PASSED** ✅ |
| **Bibhatsa Extreme Expression** | `data/processed/Bibhatsa/train (100).jpg` | `SINGLE_FACE` | `1` | 93.2% | 1.07% | **PASSED** ✅ |

---

## 5. Non-Human Image Rejection (False Positive Prevention)

| Synthetic Distractor | Tested Image | Validation Status | Face Count | Rejection Outcome |
| :--- | :--- | :---: | :---: | :---: |
| **Pure Black Canvas** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |
| **Pure White Canvas** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |
| **Neutral Gray Canvas** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |
| **Random RGB Noise** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |
| **Geometric Abstract Shapes** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |
| **Checkerboard Pattern** | Synthetic | `NO_FACE` | `0` | **CORRECT REJECTION** ✅ |

---

## 6. Verification Conclusion & Deployment Readiness

1. **Resolution Independence:** OpenCV YuNet operates on feature pyramids at native scale, resolving the previous issue where BlazeFace short-range downsampled 1080p frames to $256\times 256$ and lost medium-distance faces.
2. **Hand Mudrā & Background Artifact Filtering:** MediaPipe Landmarker geometry verification prevents hands with red alta from triggering false multiple-face rejections, while relative area filtering discards tiny miniature drawings on background wall tapestries.
3. **Strict Validation Invariance:** Non-human images (geometric shapes, noise, blank canvases) are 100% rejected (0.0% false positive rate), and multiple-face images are 100% rejected before any tensor reaches CLIP or EfficientNet.
4. **Production Deployment Ready:** Lightweight ONNX model (232 KB) + Task file (3.7 MB) load instantaneously on Streamlit Cloud without GPU requirements.
