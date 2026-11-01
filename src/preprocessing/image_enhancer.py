"""
Environmental Image Enhancer & Augmentation Pipeline for City Guardian.
Tailored for crowdsourced mobile phone captures under adverse environmental conditions:
  - Low-light sewage and night waste dumping
  - Hazy and dusty industrial environments
  - Weather variations (rain/fog) and sensor noise
  - Resilient OpenCV + PIL fallbacks
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing cv2, fall back gracefully to PIL/NumPy if cv2 is not present
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not installed — falling back to PIL/NumPy image enhancement.")


class EnvironmentalImageEnhancer:
    """
    Environmental Image Enhancement and Simulation Engine.
    Prepares citizen images for optimal YOLOv8 feature extraction.
    """

    def __init__(
        self,
        apply_clahe: bool = True,
        apply_sharpen: bool = True,
        apply_denoise: bool = False,
        sharpen_factor: float = 1.25,
    ):
        self.apply_clahe = apply_clahe
        self.apply_sharpen = apply_sharpen
        self.apply_denoise = apply_denoise
        self.sharpen_factor = sharpen_factor

    def enhance(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Runs adaptive enhancement pipeline on an input image.
        """
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if HAS_OPENCV:
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"Could not read image with cv2: {image_path}")

            if self.apply_clahe:
                img = self._apply_clahe_cv2(img)

            if self.apply_sharpen:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(self.sharpen_factor)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            save_path = output_path or str(p)
            cv2.imwrite(save_path, img)
            return save_path
        else:
            # High-fidelity PIL / NumPy fallback
            pil_img = Image.open(p).convert("RGB")
            if self.apply_clahe:
                # Adaptive histogram contrast stretch
                pil_img = ImageOps.autocontrast(pil_img, cutoff=1)

            if self.apply_sharpen:
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(self.sharpen_factor)

            save_path = output_path or str(p)
            pil_img.save(save_path, quality=95)
            return save_path

    @staticmethod
    def _apply_clahe_cv2(img: np.ndarray) -> np.ndarray:
        """Apply CLAHE to L channel in LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def adjust_low_light(pil_img: Image.Image, gamma: float = 1.35) -> Image.Image:
        """
        Adaptive gamma correction for dark sewage, unlit drains, and night dumping.
        """
        inv_gamma = 1.0 / gamma
        table = [int(((i / 255.0) ** inv_gamma) * 255) for i in range(256)]
        if pil_img.mode == "RGB":
            table = table * 3
        return pil_img.point(table)

    @staticmethod
    def simulate_atmospheric_haze(pil_img: Image.Image, haze_intensity: float = 0.25) -> Image.Image:
        """
        Simulates airborne particulate / dust haze for environmental generalization.
        """
        haze_intensity = max(0.0, min(1.0, haze_intensity))
        haze_layer = Image.new("RGB", pil_img.size, (215, 220, 225))
        return Image.blend(pil_img, haze_layer, alpha=haze_intensity * 0.35)

    @staticmethod
    def add_sensor_noise(pil_img: Image.Image, noise_sigma: float = 8.0) -> Image.Image:
        """
        Simulates low-cost mobile CMOS sensor noise in low-light captures.
        """
        arr = np.array(pil_img, dtype=np.float32)
        noise = np.random.normal(0, noise_sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


def get_domain_augmentation_hyperparameters(domain: str) -> Dict[str, Any]:
    """
    Returns domain-tuned YOLOv8 data augmentation hyperparameters (hyp.yaml)
    tailored to the visual physical realities of each environmental issue.
    """
    # Base baseline configuration
    hyp = {
        "hsv_h": 0.015,   # Mild hue jitter (avoids changing smoke/water colors)
        "hsv_s": 0.40,    # Saturation shift for wet surfaces vs dry roads
        "hsv_v": 0.35,    # Value (brightness) shift for night/shadow vs noon
        "degrees": 5.0,   # Mild rotation (mobile camera tilt)
        "translate": 0.10,# Shift
        "scale": 0.30,    # Scale variation (distance from incident)
        "shear": 2.0,     # Perspective distortion
        "perspective": 0.0005,
        "flipud": 0.0,    # Zero vertical flip (sky/ground orientation is critical)
        "fliplr": 0.5,    # Horizontal flip is geometrically valid
        "mosaic": 1.0,    # High mosaic for small clutter detection
        "mixup": 0.0,     # Default 0 to prevent unrealistic hybrid scenes
        "copy_paste": 0.0
    }

    if domain == "air_pollution":
        # Smoke and dust are amorphous; preserve hue strictly
        hyp["hsv_h"] = 0.008
        hyp["hsv_s"] = 0.25
        hyp["degrees"] = 3.0
        hyp["mosaic"] = 0.70  # Lower mosaic for large sky smoke plumes

    elif domain == "sewage_drainage":
        # Low light drains need higher brightness jitter
        hyp["hsv_v"] = 0.45
        hyp["scale"] = 0.35
        hyp["mosaic"] = 0.90

    elif domain == "solid_waste":
        # Trash has immense variety; benefit from mixup and strong mosaic
        hyp["mosaic"] = 1.0
        hyp["mixup"] = 0.15
        hyp["scale"] = 0.40

    elif domain == "water_quality":
        # Water bodies: avoid vertical flip, mild hue
        hyp["hsv_h"] = 0.010
        hyp["hsv_s"] = 0.30
        hyp["mosaic"] = 0.80

    elif domain == "mining_dust":
        # Heavy dust clouds: lower saturation variation
        hyp["hsv_s"] = 0.25
        hyp["hsv_v"] = 0.40

    return hyp


# Backwards compatibility alias
ImageEnhancer = EnvironmentalImageEnhancer

