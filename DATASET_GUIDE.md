# CITY GUARDIAN — DATASET INGESTION & ANNOTATION GUIDE

This document details dataset requirements, annotation standards, class consolidation strategies, and step-by-step ingestion procedures for training the **City Guardian** multi-domain environmental monitoring system.

---

## 1. Directory Structure Standards

All training, validation, and testing data must follow the standard Ultralytics YOLOv8 folder organization:

```text
data/
├── raw/                          # Unprocessed source downloads
│   ├── air_pollution/
│   ├── water_quality/
│   ├── sewage_drainage/
│   ├── mining_dust/
│   └── solid_waste/
├── processed/                    # Validated, deduplicated, and split datasets
│   ├── air_pollution/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── data.yaml
│   └── [other domains]/
└── quarantine/                   # Corrupted or invalid images isolated by validator
```

---

## 2. Annotation Format

Annotations must use the standard normalized YOLO text format (`.txt`), where each line represents one bounding box:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 0-indexed integer corresponding to the domain's `data.yaml` class list.
- `x_center`, `y_center`: Coordinates of bounding box center normalized to $[0.0, 1.0]$.
- `width`, `height`: Dimensions of bounding box normalized to $[0.0, 1.0]$.
- Constraint: Coordinates must satisfy $0 \le x, y, w, h \le 1.0$ and $w > 0, h > 0$.

---

## 3. Production Class Consolidation Strategy

To eliminate gradient conflict, semantic ambiguity, and extreme class imbalance, raw domain classes are consolidated into distinct non-overlapping production classes:

### Domain A: AQI & Air Pollution
*Raw candidates: 8 $\rightarrow$ Production: 4*
1. `industrial_smoke` (Merges: *industrial_emission*, *industrial_smoke*)
2. `construction_dust` (Merges: *construction_dust*, *dust_cloud*)
3. `road_dust` (Merges: *road_dust*, *vehicle_dust*)
4. `open_burning` (Merges: *open_burning*, *waste_combustion*)

### Domain B: Water Quality & Surface Pollution
*Raw candidates: 8 $\rightarrow$ Production: 4*
1. `polluted_water` (Merges: *water_discoloration*, *polluted_water*)
2. `floating_waste` (Merges: *plastic_in_water*, *floating_waste*)
3. `oil_spill` (Merges: *oil_spill*, *chemical_sheen*)
4. `sewage_discharge` (Merges: *industrial_water_discharge*, *sewage_discharge*)

### Domain C: Sewage Overflow & Drainage
*Raw candidates: 8 $\rightarrow$ Production: 4*
1. `sewage_overflow` (Merges: *overflowing_sewer*, *sewage_overflow*)
2. `blocked_drain` (Merges: *drainage_blockage*, *blocked_drain*)
3. `open_drain` (Merges: *open_drainage_channel*, *uncovered_drain*)
4. `stagnant_wastewater` (Merges: *wastewater_accumulation*, *puddle_effluent*)

### Domain D: Mining Dust & Industrial Compliance
*Raw candidates: 8 $\rightarrow$ Production: 4*
1. `mining_dust` (Merges: *exposed_mining_dust*, *quarry_dust*)
2. `uncovered_mineral_transport` (Merges: *open_truck_mineral*, *haul_truck_dust*)
3. `industrial_dust_cloud` (Merges: *blast_dust_cloud*, *crusher_plume*)
4. `construction_debris` (Merges: *excavation_rubble*, *mining_debris*)

### Domain E: Solid Waste Management & Compliance
*Raw candidates: 12 $\rightarrow$ Production: 4*
1. `plastic_waste` (Merges: *plastic_bottles*, *polybags*, *packaging*)
2. `organic_waste` (Merges: *food_scraps*, *market_waste*)
3. `illegal_dumping` (Merges: *roadside_dump*, *vacant_plot_dump*, *landfill*)
4. `overflowing_dustbin` (Merges: *spilling_container*, *full_dumpster*)

---

## 4. Ingestion & Validation Workflow

Execute the 8-point automated validation engine to check pairing, format, corruption, and cross-split leakage:

```bash
# 1. Clean and validate dataset
python scripts/validate_dataset.py --domain solid_waste

# 2. Automated group-aware splitting (70% train, 20% val, 10% test)
python scripts/prepare_dataset.py --domain solid_waste --split 0.7 0.2 0.1 --group-by-location

# 3. Perform Exploratory Data Analysis & visual inspections
python scripts/eda_analysis.py --domain solid_waste --output-dir reports/eda/solid_waste
```

---

## 5. Public Benchmark Sources & Integrated Kaggle Datasets

The repository includes direct ingestion and conversion scripts for real-world benchmark datasets:

### A. TACO (Trash Annotations in Context) — Solid Waste
- **Source**: [`kneroma/tacotrashdataset`](https://www.kaggle.com/datasets/kneroma/tacotrashdataset)
- **Scale**: 1,500 real photographic environmental images with 4,784 bounding box annotations across 60 original classes.
- **Conversion Pipeline**: Automatically converted from COCO `annotations.json` to normalized YOLOv8 `.txt` format and consolidated into the 4 non-overlapping production classes (`plastic_waste`, `organic_waste`, `illegal_dumping`, `overflowing_dustbin`).
- **Location**: Ingested into `data/raw/solid_waste/` and partitioned (350 train / 100 val / 50 test).

### B. Floating Ocean & River Waste — Water Quality
- **Source**: [`alexeydzhevello/ocean-waste-test-subset`](https://www.kaggle.com/datasets/alexeydzhevello/ocean-waste-test-subset)
- **Scale**: 91 real photographic river and lake surface images.
- **Format**: Native YOLOv8 format with bounding boxes remapped to `floating_waste` (class 1).
- **Location**: Ingested into `data/raw/water_quality/` and partitioned (63 train / 18 val / 10 test).

### C. Daily Solid Waste Municipal Time-Series — Context Baseline
- **Source**: [`ivantha/daily-solid-waste-dataset`](https://www.kaggle.com/datasets/ivantha/daily-solid-waste-dataset)
- **Scope**: Historical daily solid waste tonnage and net weights across multiple municipal councils (Boralesgamuwa, Dehiwala, Homagama, Moratuwa, Austin, Ballarat).
- **Location**: Ingested into `data/raw/solid_waste/municipal_timeseries/`.
- **Integration**: Queried via `MunicipalWasteService` (`src/context_engine/municipal_waste_service.py`) to provide statistical waste baselines for municipal wards.

---

## 6. One-Command Ingestion Pipeline

To download and ingest all datasets directly via `kagglehub`:

```bash
# 1. Download and convert all real Kaggle datasets
python scripts/ingest_kaggle_datasets.py

# 2. Partition solid waste with group-aware splitting
python scripts/prepare_dataset.py --domain solid_waste

# 3. Partition water quality with group-aware splitting
python scripts/prepare_dataset.py --domain water_quality

# 4. Generate visual EDA diagnostic charts
python scripts/eda_analysis.py --domain solid_waste
python scripts/eda_analysis.py --domain water_quality
```

