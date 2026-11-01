"""
Model Optimizer & Export Engine for City Guardian.
Supports:
  - ONNX graph export (static & dynamic batch/axes)
  - ONNX Runtime INT8 Dynamic Quantization
  - Multi-runtime inference latency profiling (PyTorch CPU vs ONNX Runtime CPU)
  - Mathematical output consistency verification (max absolute error / cosine similarity)
  - Memory and disk footprint benchmarking
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
from ultralytics import YOLO

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.warning("ONNX or onnxruntime not installed.")

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False


class ModelOptimizer:
    """
    Automates model graph export, quantization, benchmarking, and verification.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def export_to_onnx(
        self,
        model_path: str,
        output_dir: Optional[str] = None,
        dynamic: bool = True,
        simplify: bool = False,
        opset: int = 17,
        imgsz: int = 640,
    ) -> str:
        """
        Exports a PyTorch YOLOv8 checkpoint (.pt) to ONNX format.

        Args:
            model_path: Path to PyTorch .pt weights file
            output_dir: Destination folder (defaults to same folder as .pt)
            dynamic: Enable dynamic batch and spatial axes
            simplify: Run ONNX graph simplifier
            opset: ONNX operator set version (17 recommended for modern YOLOv8)
            imgsz: Input resolution square dimension

        Returns:
            Path to exported .onnx file
        """
        pt_path = Path(model_path)
        if not pt_path.exists():
            raise FileNotFoundError(f"Source model checkpoint not found: {model_path}")

        logger.info(f"Loading YOLO model for export: {model_path}")
        model = YOLO(str(pt_path))

        # Ultralytics export API handles trace and export
        logger.info(f"Exporting to ONNX (dynamic={dynamic}, opset={opset}, imgsz={imgsz})...")
        exported_path = model.export(
            format="onnx",
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            imgsz=imgsz
        )

        exported_path = Path(exported_path)

        if output_dir:
            dest_dir = Path(output_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            target_path = dest_dir / exported_path.name
            if exported_path.resolve() != target_path.resolve():
                if target_path.exists():
                    target_path.unlink()
                exported_path.rename(target_path)
                exported_path = target_path

        logger.info(f"ONNX export successful: {exported_path} ({round(os.path.getsize(exported_path) / (1024*1024), 2)} MB)")
        return str(exported_path)

    def quantize_onnx_dynamic(
        self,
        onnx_path: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Applies INT8 dynamic quantization to an exported ONNX model.
        Reduces size and improves CPU matrix multiplication throughput.
        """
        if not HAS_QUANT:
            logger.warning("onnxruntime.quantization not available; skipping quantization.")
            return None

        src_p = Path(onnx_path)
        if not src_p.exists():
            raise FileNotFoundError(f"ONNX source file not found: {onnx_path}")

        if output_path is None:
            output_path = str(src_p.with_stem(f"{src_p.stem}_int8"))

        logger.info(f"Quantizing ONNX model to INT8: {onnx_path} -> {output_path}")
        quantize_dynamic(
            model_input=str(src_p),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False
        )
        logger.info(f"Quantized model created: {output_path} ({round(os.path.getsize(output_path)/(1024*1024), 2)} MB)")
        return str(output_path)

    def benchmark_and_verify(
        self,
        pt_path: str,
        onnx_path: str,
        sample_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
        n_iterations: int = 25,
        warmup: int = 5
    ) -> Dict[str, Any]:
        """
        Compares inference latency, file sizes, and verifies prediction tensor consistency.

        Args:
            pt_path: Path to PyTorch model (.pt)
            onnx_path: Path to ONNX model (.onnx)
            sample_shape: Input tensor shape (batch, channels, height, width)
            n_iterations: Number of timed runs
            warmup: Number of warmup runs prior to timing

        Returns:
            Dictionary of metrics and comparison statistics
        """
        if not HAS_ONNX:
            raise RuntimeError("ONNX Runtime is required for benchmarking.")

        # 1. File sizes
        pt_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

        # 2. PyTorch Latency Profiling
        logger.info(f"Benchmarking PyTorch model on {self.device}...")
        yolo_pt = YOLO(pt_path)
        pt_model = yolo_pt.model
        pt_model.eval()

        dummy_np = np.random.uniform(0.0, 1.0, size=sample_shape).astype(np.float32)
        dummy_tensor = torch.from_numpy(dummy_np)

        with torch.no_grad():
            for _ in range(warmup):
                _ = pt_model(dummy_tensor)

            pt_times = []
            for _ in range(n_iterations):
                t0 = time.perf_counter()
                pt_out = pt_model(dummy_tensor)
                t1 = time.perf_counter()
                pt_times.append((t1 - t0) * 1000.0)

        # PyTorch output tensor extraction
        if isinstance(pt_out, (list, tuple)):
            pt_raw = pt_out[0].detach().cpu().numpy()
        else:
            pt_raw = pt_out.detach().cpu().numpy()

        # 3. ONNX Runtime Latency Profiling
        logger.info("Benchmarking ONNX Runtime on CPU...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        for _ in range(warmup):
            _ = session.run(None, {input_name: dummy_np})

        onnx_times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            onnx_out = session.run(None, {input_name: dummy_np})
            t1 = time.perf_counter()
            onnx_times.append((t1 - t0) * 1000.0)

        onnx_raw = onnx_out[0]

        # 4. Consistency verification
        # Compare numerical alignment between PyTorch and ONNX predictions
        shape_match = (pt_raw.shape == onnx_raw.shape)
        max_abs_err = float(np.max(np.abs(pt_raw - onnx_raw)))
        mean_abs_err = float(np.mean(np.abs(pt_raw - onnx_raw)))

        # Cosine similarity across flattened vectors
        v1 = pt_raw.flatten()
        v2 = onnx_raw.flatten()
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

        pt_mean_ms = float(np.mean(pt_times))
        onnx_mean_ms = float(np.mean(onnx_times))
        speedup = round(pt_mean_ms / max(onnx_mean_ms, 1e-6), 2)

        results = {
            "model": Path(pt_path).stem,
            "footprint": {
                "pytorch_mb": round(pt_size_mb, 2),
                "onnx_mb": round(onnx_size_mb, 2),
                "footprint_ratio": round(onnx_size_mb / max(pt_size_mb, 1e-6), 2)
            },
            "latency": {
                "pytorch_cpu": {
                    "mean_ms": round(pt_mean_ms, 2),
                    "std_ms": round(float(np.std(pt_times)), 2),
                    "p50_ms": round(float(np.percentile(pt_times, 50)), 2),
                    "p95_ms": round(float(np.percentile(pt_times, 95)), 2),
                    "fps": round(1000.0 / pt_mean_ms, 1)
                },
                "onnxruntime_cpu": {
                    "mean_ms": round(onnx_mean_ms, 2),
                    "std_ms": round(float(np.std(onnx_times)), 2),
                    "p50_ms": round(float(np.percentile(onnx_times, 50)), 2),
                    "p95_ms": round(float(np.percentile(onnx_times, 95)), 2),
                    "fps": round(1000.0 / onnx_mean_ms, 1)
                },
                "speedup_factor": f"{speedup}x"
            },
            "numerical_consistency": {
                "shape_identical": shape_match,
                "pt_shape": list(pt_raw.shape),
                "onnx_shape": list(onnx_raw.shape),
                "max_absolute_error": round(max_abs_err, 6),
                "mean_absolute_error": round(mean_abs_err, 6),
                "cosine_similarity": round(cos_sim, 6),
                "consistent": bool(max_abs_err < 1e-3 and cos_sim > 0.999)
            }
        }

        logger.info(f"Verification Results for {results['model']}:")
        logger.info(f"  PyTorch Latency: {results['latency']['pytorch_cpu']['mean_ms']} ms ({results['latency']['pytorch_cpu']['fps']} FPS)")
        logger.info(f"  ONNX Latency:    {results['latency']['onnxruntime_cpu']['mean_ms']} ms ({results['latency']['onnxruntime_cpu']['fps']} FPS) -> {speedup}x")
        logger.info(f"  Max Absolute Error: {results['numerical_consistency']['max_absolute_error']} | Cosine Sim: {results['numerical_consistency']['cosine_similarity']}")

        return results

