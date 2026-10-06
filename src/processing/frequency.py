"""
frequency.py
FFT-based frequency domain representation pipeline:
Grayscale -> Retinal mask -> Mean removal -> 2D Hann window -> 2D FFT -> Frequency shift
-> Magnitude -> Logarithmic transform -> Normalization.
"""

import cv2
import numpy as np
from typing import Tuple
from src.processing.quality import extract_retinal_mask


def create_2d_hann_window(height: int, width: int) -> np.ndarray:
    """Creates a separable 2D Hann window to taper image boundary discontinuities."""
    hann_h = np.hanning(height)
    hann_w = np.hanning(width)
    return np.outer(hann_h, hann_w).astype(np.float32)


def extract_frequency_representation(image_bgr: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Computes a numerically stable 2D FFT log-magnitude frequency representation.
    
    Pipeline:
    1. Resize image to target resolution (224, 224)
    2. Convert to grayscale
    3. Extract and apply retinal mask
    4. Subtract mean of retinal tissue to center signal around zero
    5. Apply 2D Hann window to suppress spectral leakage from rectangular aperture
    6. Compute 2D Fast Fourier Transform (FFT)
    7. Shift zero-frequency component to center (fftshift)
    8. Compute magnitude spectrum
    9. Logarithmic dynamic range compression: log(1 + magnitude)
    10. Min-Max normalization to [0.0, 1.0]
    
    Returns:
        freq_map: Float32 array of shape (H, W, 1) in range [0.0, 1.0].
    """
    if (image_bgr.shape[0], image_bgr.shape[1]) != target_size:
        image_resized = cv2.resize(image_bgr, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image_bgr

    # Grayscale conversion
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Retinal mask
    mask = extract_retinal_mask(image_resized)
    masked_pixels = gray[mask > 0]
    
    # Mean removal within retinal tissue
    if len(masked_pixels) > 0:
        mean_val = np.mean(masked_pixels)
    else:
        mean_val = np.mean(gray)
        
    zero_centered = (gray - mean_val) * (mask > 0).astype(np.float32)

    # 2D Hann window
    hann = create_2d_hann_window(target_size[0], target_size[1])
    windowed = zero_centered * hann

    # 2D FFT and shift
    fft_coeffs = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_coeffs)

    # Magnitude spectrum
    magnitude = np.abs(fft_shifted)

    # Logarithmic dynamic range compression (numerically stable)
    log_magnitude = np.log1p(magnitude)

    # Normalization to [0.0, 1.0]
    min_val = np.min(log_magnitude)
    max_val = np.max(log_magnitude)
    
    if max_val - min_val > 1e-7:
        norm_magnitude = (log_magnitude - min_val) / (max_val - min_val)
    else:
        norm_magnitude = np.zeros_like(log_magnitude)

    # Ensure finite values
    norm_magnitude = np.nan_to_num(norm_magnitude, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    return np.expand_dims(norm_magnitude, axis=-1)  # (H, W, 1)
