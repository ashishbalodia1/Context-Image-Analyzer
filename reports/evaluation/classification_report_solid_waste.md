# CITY GUARDIAN — INDIVIDUAL CLASSIFICATION REPORT
## Domain: 🗑️ Solid Waste Management & Compliance

- **Domain Identifier:** `solid_waste`
- **Model Checkpoint:** [`models/trained/solid_waste_yolov8.pt`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/models/trained/solid_waste_yolov8.pt)
- **Dataset Name:** TACO (Trash Annotations in Context) + Municipal Daily Solid Waste
- **Primary Source / Registry:** [https://www.kaggle.com/datasets/kneroma/tacotrashdataset](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)
- **Dataset Format:** COCO / YOLOv8 Real Photographic Annotations
- **Responding Authority:** Municipal Solid Waste Management Cell
- **Telemetry File:** [`reports/evaluation/classification_metrics_solid_waste.json`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/reports/evaluation/classification_metrics_solid_waste.json)

---

## 1. Overall Model Classification Performance

| Metric | Score | Percentage | Benchmark Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.815` | **81.5%** | $\ge 75.0\%$ | ✅ PASSED |
| **Precision** | `0.841` | **84.1%** | $\ge 75.0\%$ | ✅ PASSED |
| **Recall** | `0.776` | **77.6%** | $\ge 70.0\%$ | ✅ PASSED |
| **F1-Score** | `0.807` | **80.7%** | $\ge 75.0\%$ | ✅ PASSED |
| **mAP@50** | `0.812` | **81.2%** | $\ge 70.0\%$ | ✅ PASSED |
| **mAP@50-95** | `0.584` | **58.4%** | $\ge 50.0\%$ | ✅ PASSED |

---

## 2. Per-Class Detailed Breakdown Scorecard

| Violation Class | Precision | Recall | F1-Score | Specificity | Severity Weight | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `illegal_dumping` | **88.4%** | **82.5%** | **85.3%** | 94.1% | 0.85 | 120 |
| `overflowing_dustbin` | **83.5%** | **78.0%** | **80.7%** | 95.2% | 0.65 | 85 |
| `plastic_waste` | **81.2%** | **75.5%** | **78.2%** | 93.8% | 0.7 | 140 |
| `construction_debris` | **85.0%** | **76.0%** | **80.2%** | 96.0% | 0.6 | 65 |
| `organic_waste` | **82.0%** | **74.5%** | **78.1%** | 94.5% | 0.5 | 90 |
| `mixed_waste_pile` | **84.5%** | **79.0%** | **81.7%** | 95.0% | 0.75 | 110 |

---

## 3. Real-World Independent Test Benchmark (50 Held-Out TACO Photos)
- **Top-1 Classification Accuracy:** **54.0%** (27/50 real-world scenes correctly diagnosed)
- **`illegal_dumping` Detection Precision:** **70.3%** (26/37 true positives)
- **`illegal_dumping` Detection Recall:** **81.2%** (26/32 real dumps captured)
- **`illegal_dumping` F1-Score:** **75.4%**
- **Weighted Real-World F1-Score:** **51.3%**

---

## 4. Hardware Inference & Speed Profile
- **PyTorch FP32 Latency:** `89.93 ms/frame`
- **ONNX INT8 Quantized Latency:** `54.62 ms/frame`
- **Throughput:** **18.3 FPS** (Intel Core i5-12500H AVX2 SIMD)
- **Deployment Status:** Ready for real-time edge & municipal camera feed inference

---

## 5. Automated Legal & Compliance Rules
- **Illegal dumping is punishable under SWM Rules 2016** -> *GPS-tagged complaint + challan to zone officer*
