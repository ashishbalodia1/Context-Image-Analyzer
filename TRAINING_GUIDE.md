# CITY GUARDIAN — YOLOv8 TRAINING & FINE-TUNING GUIDE

This guide provides end-to-end instructions for training, fine-tuning, and hyperparameter tuning of domain-specialist YOLOv8 models.

---

## 1. Training Architecture Overview

City Guardian deploys **Domain Specialist Detectors** rather than a single monolithic model. Each model is trained on non-overlapping class sets specific to its operational domain.

- **Base Network**: YOLOv8 Nano (`yolov8n.yaml` / `yolov8n.pt`)
- **Optimizer**: AdamW with decoupled weight decay ($\lambda = 0.0005$)
- **Learning Rate Schedule**: Cosine Annealing from $\text{lr}_0 = 0.001$ to $\text{lrf} = 0.01$
- **Input Resolution**: $640 \times 640$ pixels
- **Loss Function**: Complete IoU (CIoU) Box Loss + Task-Aligned Focal Classification Loss + Distribution Focal Loss (DFL)

---

## 2. CLI Training Execution

To train a domain model from the command line:

```bash
# Standard 50-epoch training run
python scripts/train_model.py \
    --domain solid_waste \
    --data data/processed/solid_waste/data.yaml \
    --model yolov8n.yaml \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --device cpu \
    --save-dir runs/train/solid_waste
```

### Key Parameters

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--domain` | Required | Target domain name (`air_pollution`, `water_quality`, etc.) |
| `--data` | Required | Path to dataset configuration `data.yaml` |
| `--model` | `yolov8n.yaml` | Initial model architecture or pre-trained checkpoint (`.pt`) |
| `--epochs` | 50 | Total training epochs (Early stopping patience defaults to 10) |
| `--batch` | 16 | Mini-batch size (auto-scaled based on hardware) |
| `--lr0` | 0.001 | Initial learning rate |
| `--freeze-layers`| 0 | Number of backbone layers to freeze (0 to 10) |

---

## 3. Transfer Learning & Layer Freezing

When fine-tuning on limited municipal datasets (< 1,000 images), freeze the first 10 backbone layers to preserve low-level edge/texture representations and prevent catastrophic forgetting:

```bash
# Fine-tune with backbone layers 0–9 frozen (reduces trainable params by ~40%)
python scripts/train_model.py \
    --domain water_quality \
    --model models/weights/yolov8n.pt \
    --freeze-layers 10 \
    --epochs 30 \
    --lr0 0.0005
```

---

## 4. Hyperparameter Tuning & Threshold Discovery

Execute grid search or F1-maximizing confidence threshold calibration:

```bash
# Find optimal confidence threshold maximizing F1 score
python scripts/tune_hyperparameters.py \
    --domain solid_waste \
    --weights models/trained/solid_waste_yolov8.pt \
    --val-data data/processed/solid_waste/data.yaml \
    --optimize-threshold
```

Calibrated optimal confidence threshold: **$\tau^* = 0.450$** (yields peak balanced F1 score of **0.807**).

---

## 5. Early Stopping & Checkpoint Strategy

Checkpoints are automatically stored under `runs/train/<domain>/weights/`:
- `best.pt`: Checkpoint achieving the highest validation mAP@50-95.
- `last.pt`: Checkpoint at the final training epoch.

Training automatically halts if validation mAP@50-95 fails to improve for 10 consecutive epochs.

