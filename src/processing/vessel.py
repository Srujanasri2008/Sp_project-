"""
vessel.py
Retinal structural and vasculature extraction pipeline:
Fundus image -> Green channel -> Contrast enhancement -> Vessel/detail enhancement -> Retinal mask -> Vessel representation.
"""

import cv2
import numpy as np
from typing import Tuple
from src.processing.quality import extract_retinal_mask


def extract_vessel_representation(image_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extracts complementary retinal vessel structural representation.
    
    Pipeline:
    1. Resize image to working resolution (target_size)
    2. Extract green channel (highest retinal vessel contrast)
    3. Enhance local contrast with CLAHE
    4. Perform morphological top-hat filtering to isolate tubular vessel segments
    5. Mask to valid retinal FOV
    6. Normalize to [0.0, 1.0] float32 single-channel array
    
    Args:
        image_bgr: BGR fundus image.
        target_size: (H, W) resolution (default: (224, 224)).
        
    Returns:
        vessel_map: Float32 array of shape (H, W, 1) in range [0.0, 1.0].
    """
    if (image_bgr.shape[0], image_bgr.shape[1]) != target_size:
        image_resized = cv2.resize(image_bgr, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image_bgr

    # Extract retinal mask
    mask = extract_retinal_mask(image_resized)

    # Green channel has optimal hemoglobin absorption contrast
    green = image_resized[:, :, 1]

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green)

    # Multi-scale morphological vessel enhancement (vessels are darker than background)
    # Use elliptical structuring elements at small and medium scales
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    tophat_small = cv2.morphologyEx(enhanced_green, cv2.MORPH_BLACKHAT, kernel_small)
    tophat_med = cv2.morphologyEx(enhanced_green, cv2.MORPH_BLACKHAT, kernel_med)
    
    vessel_combined = cv2.addWeighted(tophat_small, 0.5, tophat_med, 0.5, 0)

    # Retinal mask application
    vessel_masked = cv2.bitwise_and(vessel_combined, vessel_combined, mask=mask)

    # Normalize to [0.0, 1.0]
    vessel_float = vessel_masked.astype(np.float32)
    max_val = np.max(vessel_float)
    if max_val > 1e-6:
        vessel_norm = vessel_float / max_val
    else:
        vessel_norm = vessel_float

    return np.expand_dims(vessel_norm, axis=-1)  # (H, W, 1)
