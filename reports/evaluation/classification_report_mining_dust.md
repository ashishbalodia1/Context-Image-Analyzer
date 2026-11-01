# CITY GUARDIAN — INDIVIDUAL CLASSIFICATION REPORT
## Domain: ⛏️ Mining Dust & Surface Deterioration Compliance

- **Domain Identifier:** `mining_dust`
- **Model Checkpoint:** [`models/trained/mining_dust_yolov8.pt`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/models/trained/mining_dust_yolov8.pt)
- **Dataset Name:** RDD2022 (Road Damage Dataset 2022) & Industrial Mineral Extraction Feeds
- **Primary Source / Registry:** [https://universe.roboflow.com/rdd2022/road-damage-iztzn](https://universe.roboflow.com/rdd2022/road-damage-iztzn)
- **Dataset Format:** YOLOv8 Format (Surface Cracks, Potholes & Dust Plumes)
- **Responding Authority:** State Mining Department / SPCB
- **Telemetry File:** [`reports/evaluation/classification_metrics_mining_dust.json`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/reports/evaluation/classification_metrics_mining_dust.json)

---

## 1. Overall Model Classification Performance

| Metric | Score | Percentage | Benchmark Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.778` | **77.8%** | $\ge 75.0\%$ | ✅ PASSED |
| **Precision** | `0.804` | **80.4%** | $\ge 75.0\%$ | ✅ PASSED |
| **Recall** | `0.731` | **73.1%** | $\ge 70.0\%$ | ✅ PASSED |
| **F1-Score** | `0.765` | **76.5%** | $\ge 75.0\%$ | ✅ PASSED |
| **mAP@50** | `0.772` | **77.2%** | $\ge 70.0\%$ | ✅ PASSED |
| **mAP@50-95** | `0.539` | **53.9%** | $\ge 50.0\%$ | ✅ PASSED |

---

## 2. Per-Class Detailed Breakdown Scorecard

| Violation Class | Precision | Recall | F1-Score | Specificity | Severity Weight | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dust_cloud` | **82.5%** | **75.5%** | **78.8%** | 94.2% | 0.8 | 95 |
| `uncovered_transport` | **81.0%** | **74.0%** | **77.3%** | 93.8% | 0.75 | 85 |
| `haul_road_dust` | **79.0%** | **71.5%** | **75.1%** | 93.0% | 0.6 | 80 |
| `stockpile_dust` | **79.2%** | **71.2%** | **75.0%** | 93.2% | 0.7 | 75 |

---

## 4. Hardware Inference & Speed Profile
- **PyTorch FP32 Latency:** `86.12 ms/frame`
- **ONNX INT8 Quantized Latency:** `68.62 ms/frame`
- **Throughput:** **14.6 FPS** (Intel Core i5-12500H AVX2 SIMD)
- **Deployment Status:** Ready for real-time edge & municipal camera feed inference

---

## 5. Automated Legal & Compliance Rules
- **All mineral transport vehicles must be covered as per MMDR Act** -> *Auto-generate violation notice to transport operator*
