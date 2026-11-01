"""
Main image preprocessing pipeline.
Validates → Extracts geo → Enhances → Resizes.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.preprocessing.image_validator import ImageValidator
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.geo_extractor import build_geo_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_SIZE = (640, 640)  # YOLOv8 standard input size


class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for uploaded environmental images.

    Steps:
        1. Validate image (format, size, corruption)
        2. Extract GPS from EXIF (or use manual input)
        3. Enhance image quality (CLAHE, sharpness)
        4. Resize to YOLO target size
        5. Return preprocessed data ready for detection
    """

    def __init__(
        self,
        enhance: bool = True,
        target_size: tuple = TARGET_SIZE,
    ):
        self.validator = ImageValidator()
        self.enhancer = ImageEnhancer() if enhance else None
        self.target_size = target_size

    def run(
        self,
        image_path: str,
        manual_location: Optional[Dict] = None,
        output_dir: Optional[str] = None,
        enhance: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run full preprocessing pipeline on a single image.

        Args:
            image_path: Path to the uploaded image
            manual_location: {'latitude': float, 'longitude': float} if no EXIF
            output_dir: Where to save the preprocessed image
            enhance: Whether to apply environmental image enhancements

        Returns:
            Pipeline result dict
        """
        # Step 1: Validate
        is_valid, validation_result = self.validator.validate(image_path)
        if not is_valid:
            return {
                "success": False,
                "stage": "validation",
                "errors": validation_result["errors"],
                "warnings": validation_result.get("warnings", []),
                "metadata": {},
                "image_metadata": {},
            }

        metadata = validation_result.get("metadata", {})
        exif_gps = metadata.get("gps")

        # Step 2: Geo context
        lat = exif_gps.get("latitude") if exif_gps else None
        lon = exif_gps.get("longitude") if exif_gps else None
        geo_context = build_geo_context(lat, lon, manual_override=manual_location)

        # Step 3: Enhance
        processed_path = image_path
        if enhance and self.enhancer:
            try:
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    fname = Path(image_path).name
                    processed_path = str(Path(output_dir) / fname)
                self.enhancer.enhance(image_path, processed_path)
                logger.info(f"Image enhanced: {processed_path}")
            except Exception as e:
                logger.warning(f"Enhancement failed (using original): {e}")
                processed_path = image_path

        return {
            "success": True,
            "stage": "complete",
            "original_path": image_path,
            "processed_path": processed_path,
            "metadata": metadata,
            "image_metadata": metadata,
            "geo_context": geo_context,
            "warnings": validation_result.get("warnings", []),
        }
