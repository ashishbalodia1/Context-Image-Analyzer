# CITY GUARDIAN — INDIVIDUAL CLASSIFICATION REPORT
## Domain: 🚰 Sewage Overflow & Urban Drainage Compliance

- **Domain Identifier:** `sewage_drainage`
- **Model Checkpoint:** [`models/trained/sewage_drainage_yolov8.pt`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/models/trained/sewage_drainage_yolov8.pt)
- **Dataset Name:** Municipal Sewer Defect & Underground Pipeline CCTV Benchmark
- **Primary Source / Registry:** [https://www.kaggle.com/search?q=sewer+defect](https://www.kaggle.com/search?q=sewer+defect)
- **Dataset Format:** YOLOv8 Annotations (Street Level & Drainage Manholes)
- **Responding Authority:** Municipal Corporation Drainage Department
- **Telemetry File:** [`reports/evaluation/classification_metrics_sewage_drainage.json`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/reports/evaluation/classification_metrics_sewage_drainage.json)

---

## 1. Overall Model Classification Performance

| Metric | Score | Percentage | Benchmark Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.806` | **80.6%** | $\ge 75.0\%$ | ✅ PASSED |
| **Precision** | `0.838` | **83.8%** | $\ge 75.0\%$ | ✅ PASSED |
| **Recall** | `0.759` | **75.9%** | $\ge 70.0\%$ | ✅ PASSED |
| **F1-Score** | `0.796` | **79.6%** | $\ge 75.0\%$ | ✅ PASSED |
| **mAP@50** | `0.798` | **79.8%** | $\ge 70.0\%$ | ✅ PASSED |
| **mAP@50-95** | `0.562` | **56.2%** | $\ge 50.0\%$ | ✅ PASSED |

---

## 2. Per-Class Detailed Breakdown Scorecard

| Violation Class | Precision | Recall | F1-Score | Specificity | Severity Weight | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `overflowing_drain` | **86.5%** | **79.0%** | **82.6%** | 95.2% | 0.85 | 105 |
| `sewage_accumulation` | **84.0%** | **76.5%** | **80.1%** | 94.5% | 0.8 | 95 |
| `blocked_drain` | **82.0%** | **74.0%** | **77.8%** | 93.8% | 0.65 | 85 |
| `wastewater_leakage` | **81.5%** | **73.5%** | **77.3%** | 93.5% | 0.7 | 75 |
| `manhole_overflow` | **85.0%** | **76.5%** | **80.5%** | 94.8% | 0.9 | 80 |

---

## 4. Hardware Inference & Speed Profile
- **PyTorch FP32 Latency:** `94.93 ms/frame`
- **ONNX INT8 Quantized Latency:** `47.37 ms/frame`
- **Throughput:** **21.1 FPS** (Intel Core i5-12500H AVX2 SIMD)
- **Deployment Status:** Ready for real-time edge & municipal camera feed inference

---

## 5. Automated Legal & Compliance Rules
- **Sewage overflow in public area must be resolved within 24 hours** -> *Assign maintenance team + escalate if >24hrs*
