"""
validation.py
Image validation module for retinal fundus images.
Performs software-level integrity, format, dimensionality, finiteness, and retinal suitability checks.
Note: Software validation does not prove or guarantee clinical diagnostic quality.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
from pathlib import Path
import cv2
import numpy as np


@dataclass
class ImageValidationResult:
    is_valid: bool
    error_message: Optional[str] = None
    image_bgr: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_fundus_image(
    image_input: Union[str, Path, bytes, np.ndarray],
    min_dimension: int = 64
) -> ImageValidationResult:
    """
    Validates an uploaded or loaded fundus image.
    
    Checks:
    1. Readable format / decodable bytes
    2. Supported extension (.jpg, .jpeg, .png) when file path given
    3. Valid shape and 3-channel structure
    4. Finite pixel values (no NaN / Inf)
    5. Non-empty content (not single-color / blank)
    6. Retinal FOV suitability check (contrast and reasonable circular FOV coverage)
    
    Returns:
        ImageValidationResult dataclass.
    """
    img_bgr = None
    file_name = None

    # Step 1: Decode input
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        file_name = path.name
        if not path.exists():
            return ImageValidationResult(False, f"File does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return ImageValidationResult(
                False,
                f"Unsupported format '{path.suffix}'. Supported formats are JPG, JPEG, PNG."
            )
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return ImageValidationResult(False, "Failed to read image file. File may be corrupted.")
            
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return ImageValidationResult(False, "Failed to decode image from byte buffer.")
            
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
    else:
        return ImageValidationResult(False, f"Unsupported image input type: {type(image_input)}")

    # Step 2: Check dimensionality & channels
    if img_bgr is None or img_bgr.size == 0:
        return ImageValidationResult(False, "Image data is empty.")

    if len(img_bgr.shape) == 2:
        # Convert grayscale to 3-channel BGR
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif len(img_bgr.shape) == 3:
        if img_bgr.shape[2] == 4:
            # Drop alpha channel
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        elif img_bgr.shape[2] != 3:
            return ImageValidationResult(
                False,
                f"Invalid channel count ({img_bgr.shape[2]}). Expected 3-channel RGB/BGR image."
            )
    else:
        return ImageValidationResult(False, f"Invalid image tensor dimensions: {img_bgr.shape}")

    h, w, c = img_bgr.shape
    if h < min_dimension or w < min_dimension:
        return ImageValidationResult(
            False,
            f"Image dimensions ({w}x{h}) are too small. Minimum required dimension is {min_dimension}x{min_dimension}."
        )

    # Step 3: Check finiteness
    if not np.all(np.isfinite(img_bgr)):
        return ImageValidationResult(False, "Image contains non-finite pixel values (NaN or Inf).")

    # Step 4: Check non-empty / non-flat content
    std_val = float(np.std(img_bgr))
    if std_val < 1.0:
        return ImageValidationResult(
            False,
            "Image is uniform or near-blank (intensity standard deviation < 1.0)."
        )

    # Step 5: Retinal FOV suitability check
    # Fundus images have an illuminated retinal field against dark or framed background
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fov_ratio = float(np.count_nonzero(binary)) / float(h * w)

    if fov_ratio < 0.10:
        return ImageValidationResult(
            False,
            "Image lacks sufficient illuminated retinal area (< 10% non-dark field of view)."
        )

    metadata = {
        "width": w,
        "height": h,
        "channels": c,
        "intensity_mean": round(float(np.mean(img_bgr)), 2),
        "intensity_std": round(std_val, 2),
        "fov_ratio": round(fov_ratio, 3),
        "file_name": file_name
    }

    return ImageValidationResult(
        is_valid=True,
        error_message=None,
        image_bgr=img_bgr,
        metadata=metadata
    )
