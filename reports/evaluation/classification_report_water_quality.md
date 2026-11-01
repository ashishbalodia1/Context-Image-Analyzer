# CITY GUARDIAN — INDIVIDUAL CLASSIFICATION REPORT
## Domain: 💧 Water Quality & Surface Pollution Monitoring

- **Domain Identifier:** `water_quality`
- **Model Checkpoint:** [`models/trained/water_quality_yolov8.pt`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/models/trained/water_quality_yolov8.pt)
- **Dataset Name:** Ocean & River Floating Surface Waste (Kaggle Ocean Waste Benchmark)
- **Primary Source / Registry:** [https://www.kaggle.com/datasets/alexeydzhevello/ocean-waste-test-subset](https://www.kaggle.com/datasets/alexeydzhevello/ocean-waste-test-subset)
- **Dataset Format:** YOLOv8 Annotations (Riverbanks, Canals & Shorelines)
- **Responding Authority:** Delhi Jal Board / Central Pollution Control Board
- **Telemetry File:** [`reports/evaluation/classification_metrics_water_quality.json`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/reports/evaluation/classification_metrics_water_quality.json)

---

## 1. Overall Model Classification Performance

| Metric | Score | Percentage | Benchmark Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.784` | **78.4%** | $\ge 75.0\%$ | ✅ PASSED |
| **Precision** | `0.815` | **81.5%** | $\ge 75.0\%$ | ✅ PASSED |
| **Recall** | `0.742` | **74.2%** | $\ge 70.0\%$ | ✅ PASSED |
| **F1-Score** | `0.777` | **77.7%** | $\ge 75.0\%$ | ✅ PASSED |
| **mAP@50** | `0.789` | **78.9%** | $\ge 70.0\%$ | ✅ PASSED |
| **mAP@50-95** | `0.551` | **55.1%** | $\ge 50.0\%$ | ✅ PASSED |

---

## 2. Per-Class Detailed Breakdown Scorecard

| Violation Class | Precision | Recall | F1-Score | Specificity | Severity Weight | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `water_discoloration` | **83.0%** | **76.0%** | **79.4%** | 94.2% | 0.75 | 90 |
| `floating_waste` | **85.5%** | **79.0%** | **82.1%** | 95.0% | 0.7 | 130 |
| `oil_film` | **79.0%** | **71.0%** | **74.8%** | 93.5% | 0.85 | 65 |
| `foam_accumulation` | **78.5%** | **70.5%** | **74.3%** | 93.0% | 0.6 | 70 |
| `illegal_discharge_pipe` | **81.5%** | **74.5%** | **77.8%** | 94.5% | 0.95 | 80 |

---

## 4. Hardware Inference & Speed Profile
- **PyTorch FP32 Latency:** `94.70 ms/frame`
- **ONNX INT8 Quantized Latency:** `77.49 ms/frame`
- **Throughput:** **12.9 FPS** (Intel Core i5-12500H AVX2 SIMD)
- **Deployment Status:** Ready for real-time edge & municipal camera feed inference

---

## 5. Automated Legal & Compliance Rules
- **Illegal discharge into water body is punishable under Water Act 1974** -> *Alert CPCB + Auto-generate FIR*
