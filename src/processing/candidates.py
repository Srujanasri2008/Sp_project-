"""
candidates.py
Implementation of candidate DSP processing paths and comparator adaptive enhancement:
- PATH A: LAB color space + L-channel CLAHE (clipLimit=1.5, tileGridSize=(8,8))
- PATH B: Gaussian Blur 3x3 + Path A
- PATH C: Illumination correction (sigma=15, floor=0.05, mean scaling) + Path A
- AGCWD: Adaptive Gamma Correction with Weighting Distribution (Literature comparator)
"""

import cv2
import numpy as np
from typing import Dict, Any


def apply_path_a(image_bgr: np.ndarray, clip_limit: float = 1.5, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    PATH A:
    Transforms image to LAB color space, applies CLAHE to the L (luminance) channel,
    and converts back to BGR color space.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_clahe, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_path_b(image_bgr: np.ndarray, clip_limit: float = 1.5, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    PATH B:
    Applies a 3x3 Gaussian blur for mild noise suppression, followed by Path A.
    """
    blurred = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    return apply_path_a(blurred, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


def apply_illumination_correction(image_bgr: np.ndarray, sigma: float = 15.0, floor: float = 0.05) -> np.ndarray:
    """
    Illumination correction via low-frequency background estimation and normalization:
    I_corr = (I / (Gaussian_sigma(I) + floor)) * mean(Gaussian_sigma(I))
    """
    img_float = image_bgr.astype(np.float32) / 255.0
    # Estimate illumination background via Gaussian blur
    # When ksize is (0,0), OpenCV calculates kernel size from sigma
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    mean_val = float(np.mean(blur))
    corrected = (img_float / (blur + floor)) * mean_val
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return corrected


def apply_path_c(image_bgr: np.ndarray, sigma: float = 15.0, floor: float = 0.05,
                 clip_limit: float = 1.5, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    PATH C:
    Applies illumination correction followed by Path A.
    """
    corrected = apply_illumination_correction(image_bgr, sigma=sigma, floor=floor)
    return apply_path_a(corrected, clip_limit=clip_limit, tile_grid_size=tile_grid_size)


def apply_agcwd(image_bgr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Adaptive Gamma Correction with Weighting Distribution (AGCWD) - Literature comparator.
    Huang et al., IEEE TIP 2013.
    Applied to the Luminance (L) channel in LAB space to preserve natural retinal colorimetry.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate probability density function (PDF)
    hist, _ = np.histogram(l.flatten(), bins=256, range=[0, 256], density=True)
    
    # Weighting distribution
    p_max = np.max(hist)
    p_min = np.min(hist)
    
    if p_max - p_min > 1e-7:
        p_w = p_max * (((hist - p_min) / (p_max - p_min)) ** alpha)
    else:
        p_w = hist
        
    # Cumulative distribution function (CDF)
    sum_pw = np.sum(p_w)
    if sum_pw > 1e-7:
        c_w = np.cumsum(p_w) / sum_pw
    else:
        c_w = np.linspace(0, 1, 256)
        
    # Gamma mapping table: T(l) = 255 * (l / 255) ** (1 - c_w(l))
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        gamma = 1.0 - c_w[i]
        val = 255.0 * ((i / 255.0) ** gamma)
        lookup[i] = int(np.clip(val, 0, 255))
        
    l_agcwd = cv2.LUT(l, lookup)
    lab_enhanced = cv2.merge([l_agcwd, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def generate_candidate_paths(image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generates all three candidate DSP paths for a fundus image.
    
    Returns:
        Dictionary mapping path names ('PATH_A', 'PATH_B', 'PATH_C') to enhanced BGR images.
    """
    return {
        "PATH_A": apply_path_a(image_bgr),
        "PATH_B": apply_path_b(image_bgr),
        "PATH_C": apply_path_c(image_bgr)
    }
