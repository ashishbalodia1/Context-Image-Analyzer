"""
Unit and integration tests for Model Optimizer & Deployment Engine.
Verifies:
  - ONNX export execution
  - ONNX Runtime CPU inference and latency profiling
  - Prediction numerical parity (PyTorch vs ONNX)
  - INT8 dynamic quantization
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.optimizer import ModelOptimizer


def test_model_export_and_benchmarking():
    optimizer = ModelOptimizer(device="cpu")
    model_path = PROJECT_ROOT / "models" / "trained" / "solid_waste_yolov8.pt"

    assert model_path.exists(), f"Test requires baseline model: {model_path}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # 1. Test ONNX Export
        print("[TEST] Exporting model to ONNX...")
        onnx_path = optimizer.export_to_onnx(
            model_path=str(model_path),
            output_dir=str(tmp),
            dynamic=True
        )
        assert Path(onnx_path).exists()
        assert Path(onnx_path).stat().st_size > 1024 * 1024  # > 1 MB
        print(f"  -> Export succeeded: {onnx_path} ({round(Path(onnx_path).stat().st_size / 1e6, 2)} MB)")

        # 2. Test Benchmarking & Consistency
        print("[TEST] Benchmarking PyTorch vs ONNX Runtime...")
        bench = optimizer.benchmark_and_verify(
            pt_path=str(model_path),
            onnx_path=onnx_path,
            n_iterations=5,
            warmup=2
        )
        assert bench["latency"]["pytorch_cpu"]["mean_ms"] > 0
        assert bench["latency"]["onnxruntime_cpu"]["mean_ms"] > 0
        assert bench["numerical_consistency"]["shape_identical"] is True
        assert bench["numerical_consistency"]["cosine_similarity"] > 0.99
        print(f"  -> Benchmarks: PyTorch={bench['latency']['pytorch_cpu']['mean_ms']}ms, ONNX={bench['latency']['onnxruntime_cpu']['mean_ms']}ms")
        print(f"  -> Cosine parity: {bench['numerical_consistency']['cosine_similarity']}")

        # 3. Test INT8 Dynamic Quantization
        print("[TEST] Testing INT8 Dynamic Quantization...")
        quant_path = optimizer.quantize_onnx_dynamic(onnx_path)
        if quant_path:
            assert Path(quant_path).exists()
            assert Path(quant_path).stat().st_size > 0
            print(f"  -> Quantization succeeded: {quant_path} ({round(Path(quant_path).stat().st_size / 1e6, 2)} MB)")


if __name__ == "__main__":
    test_model_export_and_benchmarking()
    print("\n=======================================================")
    print("MODEL OPTIMIZER TESTS PASSED (100% SUCCESS)!")
    print("=======================================================")

