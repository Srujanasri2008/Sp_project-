"""
dataset.py
PyTorch Dataset and DataLoader definitions for the APTOS Diabetic Retinopathy dataset.
Supports official splits (train/val/test), on-demand candidate path generation,
ARPS adaptive selection, vessel structural representation, and FFT spectral extraction.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.processing.arps import select_processing_path
from src.processing.candidates import apply_path_a, apply_path_b, apply_path_c, apply_agcwd
from src.processing.vessel import extract_vessel_representation
from src.processing.frequency import extract_frequency_representation

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARPS_CACHE_PATH = PROJECT_ROOT / "data" / "metadata" / "arps_selection_cache.csv"


class RetinalFundusDataset(Dataset):
    """
    Dataset loader for retinal fundus images with multi-domain signal processing support.
    
    Modes:
        - "proposed_arps": Evaluates candidates, applies image-specific ARPS path, 
                           extracts vessel and frequency domains.
        - "baseline": Applies fixed standard preprocessing (Path A / LAB CLAHE).
        - "agcwd": Applies literature comparator adaptive gamma correction.
        - "raw": Resized normalized RGB image without DSP enhancements.
    """
    def __init__(
        self,
        manifest_path: str = "data/splits/official_split_manifest.csv",
        split: str = "train",
        mode: str = "proposed_arps",
        include_vessel: bool = True,
        include_frequency: bool = True,
        transform = None
    ):
        super().__init__()
        self.manifest_path = PROJECT_ROOT / manifest_path
        self.split = split
        self.mode = mode
        self.include_vessel = include_vessel
        self.include_frequency = include_frequency
        self.transform = transform
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
            
        full_df = pd.read_csv(self.manifest_path)
        self.df = full_df[full_df["split"] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in manifest.")

        # Load offline ARPS selection cache if available
        self.arps_cache: Dict[str, Dict[str, Any]] = {}
        if ARPS_CACHE_PATH.exists():
            cache_df = pd.read_csv(ARPS_CACHE_PATH)
            for _, r in cache_df.iterrows():
                self.arps_cache[r["image_id"]] = {
                    "selected_path": r["selected_path"],
                    "arps_score": float(r["arps_score"])
                }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image_path = PROJECT_ROOT / row["file_path"]
        
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise IOError(f"Could not read fundus image at {image_path}")

        # Ensure working resolution is 224x224
        if image_bgr.shape[:2] != (224, 224):
            image_bgr = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        selected_path_name = "NONE"
        arps_score = 0.0

        # Processing domain selection
        if self.mode == "proposed_arps":
            if image_id in self.arps_cache:
                cached = self.arps_cache[image_id]
                selected_path_name = cached["selected_path"]
                arps_score = cached["arps_score"]
                if selected_path_name == "PATH_A":
                    proc_bgr = apply_path_a(image_bgr)
                elif selected_path_name == "PATH_B":
                    proc_bgr = apply_path_b(image_bgr)
                elif selected_path_name == "PATH_C":
                    proc_bgr = apply_path_c(image_bgr)
                else:
                    proc_bgr = apply_path_a(image_bgr)
            else:
                arps_res = select_processing_path(image_bgr)
                proc_bgr = arps_res.selected_image
                selected_path_name = arps_res.selected_path
                arps_score = arps_res.arps_score
        elif self.mode == "baseline":
            proc_bgr = apply_path_a(image_bgr)
            selected_path_name = "PATH_A"
        elif self.mode == "agcwd":
            proc_bgr = apply_agcwd(image_bgr)
            selected_path_name = "AGCWD"
        elif self.mode == "raw":
            proc_bgr = image_bgr
            selected_path_name = "RAW"
        else:
            raise ValueError(f"Unknown processing mode: {self.mode}")

        # Convert to RGB float32 tensor in range [0, 1]
        proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Shape: (3, 224, 224)
        spatial_tensor = torch.from_numpy(proc_rgb).permute(2, 0, 1)

        # Vessel domain representation (1, 224, 224)
        if self.include_vessel:
            vessel_map = extract_vessel_representation(proc_bgr, target_size=(224, 224))
            vessel_tensor = torch.from_numpy(vessel_map).permute(2, 0, 1)
        else:
            vessel_tensor = torch.empty(0)

        # Frequency domain representation (1, 224, 224)
        if self.include_frequency:
            freq_map = extract_frequency_representation(proc_bgr, target_size=(224, 224))
            freq_tensor = torch.from_numpy(freq_map).permute(2, 0, 1)
        else:
            freq_tensor = torch.empty(0)

        binary_label = torch.tensor(int(row["binary_label"]), dtype=torch.long)
        five_class_label = torch.tensor(int(row["five_class"]), dtype=torch.long)

        return {
            "image_id": row["image_id"],
            "spatial": spatial_tensor,
            "vessel": vessel_tensor,
            "frequency": freq_tensor,
            "binary_label": binary_label,
            "five_class_label": five_class_label,
            "selected_path": selected_path_name,
            "arps_score": arps_score
        }


def create_dataloader(
    manifest_path: str = "data/splits/official_split_manifest.csv",
    split: str = "train",
    mode: str = "proposed_arps",
    include_vessel: bool = True,
    include_frequency: bool = True,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Creates a configured DataLoader for training or evaluation."""
    dataset = RetinalFundusDataset(
        manifest_path=manifest_path,
        split=split,
        mode=mode,
        include_vessel=include_vessel,
        include_frequency=include_frequency
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False
    )
