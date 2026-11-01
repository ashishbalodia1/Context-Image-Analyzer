# CITY GUARDIAN — INDIVIDUAL CLASSIFICATION REPORT
## Domain: 🏭 AQI & Air Pollution Monitoring

- **Domain Identifier:** `air_pollution`
- **Model Checkpoint:** [`models/trained/air_pollution_yolov8.pt`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/models/trained/air_pollution_yolov8.pt)
- **Dataset Name:** FASDD (Fire & Smoke Dataset CV) & D-Fire Real Plume Benchmark
- **Primary Source / Registry:** [https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv](https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv)
- **Dataset Format:** YOLOv8 Annotations (Aerial + Ground Camera Feeds)
- **Responding Authority:** Delhi Pollution Control Committee (DPCC)
- **Telemetry File:** [`reports/evaluation/classification_metrics_air_pollution.json`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/reports/evaluation/classification_metrics_air_pollution.json)

---

## 1. Overall Model Classification Performance

| Metric | Score | Percentage | Benchmark Gate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | `0.802` | **80.2%** | $\ge 75.0\%$ | ✅ PASSED |
| **Precision** | `0.829` | **82.9%** | $\ge 75.0\%$ | ✅ PASSED |
| **Recall** | `0.768` | **76.8%** | $\ge 70.0\%$ | ✅ PASSED |
| **F1-Score** | `0.797` | **79.7%** | $\ge 75.0\%$ | ✅ PASSED |
| **mAP@50** | `0.805` | **80.5%** | $\ge 70.0\%$ | ✅ PASSED |
| **mAP@50-95** | `0.573` | **57.3%** | $\ge 50.0\%$ | ✅ PASSED |

---

## 2. Per-Class Detailed Breakdown Scorecard

| Violation Class | Precision | Recall | F1-Score | Specificity | Severity Weight | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `industrial_smoke` | **86.2%** | **81.0%** | **83.5%** | 95.5% | 0.9 | 115 |
| `open_burning` | **84.8%** | **79.5%** | **82.1%** | 94.8% | 0.85 | 95 |
| `construction_dust` | **81.5%** | **74.0%** | **77.6%** | 93.2% | 0.6 | 80 |
| `road_dust` | **79.5%** | **72.5%** | **75.8%** | 92.5% | 0.4 | 75 |
| `vehicle_exhaust` | **82.5%** | **77.0%** | **79.7%** | 94.0% | 0.7 | 85 |

---

## 4. Hardware Inference & Speed Profile
- **PyTorch FP32 Latency:** `94.96 ms/frame`
- **ONNX INT8 Quantized Latency:** `70.43 ms/frame`
- **Throughput:** **14.2 FPS** (Intel Core i5-12500H AVX2 SIMD)
- **Deployment Status:** Ready for real-time edge & municipal camera feed inference

---

## 5. Automated Legal & Compliance Rules
- **Open burning within 5km of residential area is a violation** -> *Immediate FIR + department alert*
