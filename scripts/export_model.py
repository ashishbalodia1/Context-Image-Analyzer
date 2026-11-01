"""
CLI Model Export and Optimization Script for City Guardian.
Exports PyTorch weights (.pt) to ONNX, quantizes to INT8, benchmarks runtimes,
and verifies numerical prediction parity.

Usage:
  python scripts/export_model.py --domain solid_waste
  python scripts/export_model.py --all-domains
  python scripts/export_model.py --model models/trained/solid_waste_yolov8.pt --quantize
"""
import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.detection.optimizer import ModelOptimizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

ALL_DOMAINS = [
    "air_pollution",
    "water_quality",
    "sewage_drainage",
    "mining_dust",
    "solid_waste",
]


def main():
    parser = argparse.ArgumentParser(description="Export and optimize YOLOv8 models for production.")
    parser.add_argument("--model", type=str, help="Path to a specific .pt model file")
    parser.add_argument("--domain", type=str, choices=ALL_DOMAINS, help="Specific domain model to export")
    parser.add_argument("--all-domains", action="store_true", help="Export and benchmark all 5 environmental domain models")
    parser.add_argument("--output-dir", type=str, default="models/optimized", help="Directory for exported models")
    parser.add_argument("--quantize", action="store_true", help="Generate dynamic INT8 quantized ONNX models")
    parser.add_argument("--benchmark", action="store_true", default=True, help="Benchmark PyTorch vs ONNX latency and verify parity")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmarking iterations")
    parser.add_argument("--report-dir", type=str, default="reports/optimization", help="Output directory for benchmark reports")

    args = parser.parse_args()
    optimizer = ModelOptimizer(device="cpu")

    targets = []
    if args.all_domains:
        for dom in ALL_DOMAINS:
            p = PROJECT_ROOT / "models" / "trained" / f"{dom}_yolov8.pt"
            if p.exists():
                targets.append((dom, str(p)))
            else:
                logger.warning(f"Domain model checkpoint not found: {p}")
    elif args.domain:
        p = PROJECT_ROOT / "models" / "trained" / f"{args.domain}_yolov8.pt"
        targets.append((args.domain, str(p)))
    elif args.model:
        p = Path(args.model)
        targets.append((p.stem, str(p)))
    else:
        # Default to solid_waste
        p = PROJECT_ROOT / "models" / "trained" / "solid_waste_yolov8.pt"
        targets.append(("solid_waste", str(p)))

    if not targets:
        logger.error("No valid model targets found for export.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, pt_path in targets:
        print(f"\n=======================================================")
        print(f" OPTIMIZING: {name} ({pt_path})")
        print(f"=======================================================")

        # 1. Export ONNX
        onnx_file = optimizer.export_to_onnx(
            model_path=pt_path,
            output_dir=str(out_dir),
            dynamic=True,
            simplify=False
        )
        print(f"[EXPORT] Successfully generated ONNX: {onnx_file}")

        quant_file = None
        if args.quantize:
            quant_file = optimizer.quantize_onnx_dynamic(onnx_file)
            print(f"[QUANTIZE] Generated INT8 ONNX: {quant_file}")

        # 2. Benchmark & Verify
        if args.benchmark:
            print("[BENCHMARK] Measuring CPU latency and numerical parity...")
            bench_res = optimizer.benchmark_and_verify(
                pt_path=pt_path,
                onnx_path=onnx_file,
                n_iterations=args.iterations
            )
            bench_res["quantized_onnx"] = quant_file
            all_results[name] = bench_res

            print("\n  Runtime Latency Summary:")
            print(f"   - PyTorch CPU:       {bench_res['latency']['pytorch_cpu']['mean_ms']} ms ({bench_res['latency']['pytorch_cpu']['fps']} FPS)")
            print(f"   - ONNX Runtime CPU:  {bench_res['latency']['onnxruntime_cpu']['mean_ms']} ms ({bench_res['latency']['onnxruntime_cpu']['fps']} FPS)")
            print(f"   - Speedup Factor:    {bench_res['latency']['speedup_factor']}")
            print(f"   - Numerical Parity:  Consistent={bench_res['numerical_consistency']['consistent']} (Max Abs Err={bench_res['numerical_consistency']['max_absolute_error']})")

    # Save summary report
    report_path = report_dir / "optimization_benchmarks.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[DONE] Benchmark report written to {report_path}")


if __name__ == "__main__":
    main()

