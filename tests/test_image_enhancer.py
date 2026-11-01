"""
Unit tests for EnvironmentalImageEnhancer and domain augmentation profiles.
"""
import sys
import tempfile
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.image_enhancer import EnvironmentalImageEnhancer, get_domain_augmentation_hyperparameters


def test_environmental_enhancer_methods():
    enhancer = EnvironmentalImageEnhancer()
    im = Image.new("RGB", (100, 100), color=(30, 30, 30))  # Dark image

    # 1. Low light adjustment
    brightened = enhancer.adjust_low_light(im, gamma=1.5)
    orig_pixel = im.getpixel((50, 50))[0]
    new_pixel = brightened.getpixel((50, 50))[0]
    assert new_pixel > orig_pixel, "Gamma adjustment should increase luminance in dark images"

    # 2. Haze simulation
    hazed = enhancer.simulate_atmospheric_haze(im, haze_intensity=0.3)
    assert hazed.size == im.size
    assert hazed.mode == "RGB"

    # 3. Sensor noise
    noisy = enhancer.add_sensor_noise(im, noise_sigma=5.0)
    assert noisy.size == im.size
    assert noisy.mode == "RGB"


def test_domain_augmentation_profiles():
    # Air pollution: lower mosaic, strict hue preservation
    air_hyp = get_domain_augmentation_hyperparameters("air_pollution")
    assert air_hyp["hsv_h"] < 0.01
    assert air_hyp["flipud"] == 0.0  # Smoke never flows downward

    # Solid waste: high mosaic and mixup
    waste_hyp = get_domain_augmentation_hyperparameters("solid_waste")
    assert waste_hyp["mosaic"] == 1.0
    assert waste_hyp["mixup"] == 0.15

    # Sewage: high brightness range for night/dark drains
    sewage_hyp = get_domain_augmentation_hyperparameters("sewage_drainage")
    assert sewage_hyp["hsv_v"] >= 0.40


if __name__ == "__main__":
    test_environmental_enhancer_methods()
    test_domain_augmentation_profiles()
    print("IMAGE ENHANCER & AUGMENTATION TESTS PASSED (100% SUCCESS)!")

