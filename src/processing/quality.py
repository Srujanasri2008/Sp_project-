"""
quality.py
Image signal and quality analysis module for retinal fundus images.
Computes contrast, sharpness, noise indicators, illumination variation,
and vessel visibility with consistent retinal masking.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


def extract_retinal_mask(image_bgr: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Extracts a binary mask representing the retinal field of view (FOV).
    Eliminates background black borders to prevent distorting quality metrics.
    
    Args:
        image_bgr: Input BGR image (H, W, 3).
        threshold: Intensity threshold to separate retina from black surround.
        
    Returns:
        Binary mask (H, W) where 1 indicates retinal tissue, 0 background.
    """
    # Use grayscale or maximum across channels
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Fill small holes inside the retina
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Keep only largest connected component (the retinal disc)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Label 0 is background; find largest non-zero component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        retina_mask = (labels == largest_idx).astype(np.uint8)
    else:
        retina_mask = (binary > 0).astype(np.uint8)
        
    # Erode slightly (3 pixels) to avoid boundary edge contrast artifacts
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    retina_mask = cv2.erode(retina_mask, erode_kernel)
    
    # If mask is degenerately small, fall back to full image
    if np.sum(retina_mask) < 100:
        retina_mask = np.ones(gray.shape, dtype=np.uint8)
        
    return retina_mask


def compute_contrast(gray: np.ndarray, mask: np.ndarray) -> float:
    """Computes RMS contrast (standard deviation of pixel values) within retinal mask."""
    pixels = gray[mask > 0]
    if len(pixels) == 0:
        return 0.0
    return float(np.std(pixels))


def compute_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """Computes sharpness using the variance of the Laplacian within the retinal mask."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    # Erode mask slightly more for Laplacian to avoid any outer boundary edge step
    lap_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    pixels = laplacian[lap_mask > 0]
    if len(pixels) == 0:
        return 0.0
    return float(np.var(pixels))


def compute_noise_indicator(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes a robust high-frequency noise estimate using Median Absolute Deviation (MAD)
    of the difference between image and a 3x3 median-filtered version.
    """
    med = cv2.medianBlur(gray, 3)
    diff = np.abs(gray.astype(np.float64) - med.astype(np.float64))
    pixels = diff[mask > 0]
    if len(pixels) == 0:
        return 0.0
    # Standard normal MAD estimator constant: 1 / 0.6745 ≈ 1.4826
    mad = np.median(pixels) * 1.4826
    return float(mad)


def compute_illumination_variation(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Estimates illumination variation via the standard deviation of a low-pass
    Gaussian blurred version of the image inside the retinal FOV.
    """
    # Large Gaussian blur captures low-frequency illumination profile
    low_freq = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 15.0)
    pixels = low_freq[mask > 0]
    if len(pixels) == 0:
        return 0.0
    mean_val = np.mean(pixels) + 1e-5
    std_val = np.std(pixels)
    # Coefficient of variation (relative illumination heterogeneity) * 100
    return float((std_val / mean_val) * 100.0)


def compute_vessel_visibility(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Extracts retinal vessel structural information from the green channel
    and calculates vessel visibility metric.
    
    Returns:
        vessel_metric: Scalar structural energy / density value.
        vessel_map: 2D uint8 vessel response map.
    """
    # Green channel has the highest contrast for retinal vasculature
    green = image_bgr[:, :, 1]
    
    # Morphological Top-Hat to emphasize tubular vessel structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_hat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    
    # CLAHE on green channel to normalize background
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green)
    
    # Black top-hat (vessels are darker than surrounding retinal background in green)
    black_hat = cv2.morphologyEx(enhanced_green, cv2.MORPH_BLACKHAT, kernel)
    
    # Combined vessel response masked to retina
    vessel_resp = cv2.bitwise_and(black_hat, black_hat, mask=mask)
    
    pixels = vessel_resp[mask > 0]
    vessel_metric = float(np.mean(pixels)) if len(pixels) > 0 else 0.0
    return vessel_metric, vessel_resp


def analyze_image_quality(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Performs full signal/quality analysis on a fundus image.
    
    Args:
        image_bgr: BGR fundus image.
        
    Returns:
        Dictionary containing:
        - contrast
        - sharpness
        - noise
        - illumination_variation
        - vessel_visibility
        - mask_coverage_ratio
    """
    mask = extract_retinal_mask(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    contrast = compute_contrast(gray, mask)
    sharpness = compute_sharpness(gray, mask)
    noise = compute_noise_indicator(gray, mask)
    illumination = compute_illumination_variation(gray, mask)
    vessel_metric, _ = compute_vessel_visibility(image_bgr, mask)
    
    total_pixels = mask.shape[0] * mask.shape[1]
    coverage = float(np.sum(mask) / total_pixels)
    
    return {
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "noise": round(noise, 4),
        "illumination_variation": round(illumination, 4),
        "vessel_visibility": round(vessel_metric, 4),
        "mask_coverage": round(coverage, 4)
    }
