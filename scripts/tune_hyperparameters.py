"""
CLI Runner for Systematic Hyperparameter Optimization & Model Comparison.
Usage:
  python scripts/tune_hyperparameters.py --domain solid_waste --data-yaml models/configs/solid_waste.yaml --compare-optimizers
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from src.detection.tuner import HyperparameterTuner


def main():
    parser = argparse.ArgumentParser(description="Systematic Hyperparameter Tuning for City Guardian")
    parser.add_argument("--domain", type=str, required=True,
                        choices=["air_pollution", "water_quality", "sewage_drainage", "mining_dust", "solid_waste"])
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--compare-optimizers", action="store_true", help="Compare AdamW vs SGD")
    parser.add_argument("--compare-backbone-freeze", action="store_true", help="Compare frozen backbone vs full fine-tuning")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    tuner = HyperparameterTuner()

    if args.compare_optimizers:
        print(f"Executing Optimizer Comparison (AdamW vs SGD) on domain [{args.domain}]...")
        # Trial 1: AdamW
        tuner.run_controlled_experiment(
            exp_id=f"{args.domain}_adamw",
            domain=args.domain,
            data_yaml=args.data_yaml,
            optimizer="AdamW",
            lr0=0.001,
            epochs=args.epochs,
            batch=args.batch
        )
        # Trial 2: SGD with momentum
        tuner.run_controlled_experiment(
            exp_id=f"{args.domain}_sgd",
            domain=args.domain,
            data_yaml=args.data_yaml,
            optimizer="SGD",
            lr0=0.01,
            epochs=args.epochs,
            batch=args.batch
        )

    elif args.compare_backbone_freeze:
        print(f"Executing Backbone Freezing Comparison on domain [{args.domain}]...")
        # Trial 1: Full fine-tuning
        tuner.run_controlled_experiment(
            exp_id=f"{args.domain}_full_ft",
            domain=args.domain,
            data_yaml=args.data_yaml,
            freeze_backbone=False,
            epochs=args.epochs,
            batch=args.batch
        )
        # Trial 2: Frozen backbone (first 10 layers)
        tuner.run_controlled_experiment(
            exp_id=f"{args.domain}_freeze10",
            domain=args.domain,
            data_yaml=args.data_yaml,
            freeze_backbone=True,
            freeze_layers=10,
            epochs=args.epochs,
            batch=args.batch
        )

    tuner.print_comparison_table()


if __name__ == "__main__":
    main()

