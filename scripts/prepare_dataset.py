"""
prepare_dataset.py
Downloads and standardizes the bumbledeep/aptos dataset (3,662 images).
Resizes to working resolution (224x224), maps binary and five-class labels,
and creates the deterministic official stratified split:
- Train: 2563
- Validation: 549
- Test: 550
Saves official manifest and split indices.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


def prepare_aptos_dataset():
    print("=" * 65)
    print("PHASE 2: DATASET INGESTION & DETERMINISTIC SPLIT")
    print("=" * 65)
    
    images_dir = PROJECT_ROOT / "data" / "raw" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = PROJECT_ROOT / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Loading 'bumbledeep/aptos' dataset from Hugging Face...")
    ds = load_dataset("bumbledeep/aptos", split="train")
    total_samples = len(ds)
    print(f"  Loaded {total_samples} samples (Expected: 3662).")
    assert total_samples == 3662, f"Expected 3662 images, found {total_samples}"

    print("[2/5] Standardizing images to 224x224 and saving metadata...")
    records = []
    
    for idx, item in enumerate(tqdm(ds, desc="Processing images")):
        image_id = f"aptos_{idx:04d}"
        label_code = int(item["label_code"])
        label_name = CLASS_NAMES.get(label_code, "Unknown")
        binary_label = 0 if label_code == 0 else 1
        
        rel_file_path = f"data/raw/images/{image_id}.png"
        abs_file_path = images_dir / f"{image_id}.png"
        
        # Save image at 224x224 if not already present
        if not abs_file_path.exists():
            pil_img = item["image"].convert("RGB")
            pil_img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
            pil_img_resized.save(abs_file_path, "PNG")
            
        records.append({
            "image_id": image_id,
            "file_path": rel_file_path,
            "five_class": label_code,
            "class_name": label_name,
            "binary_label": binary_label
        })

    df = pd.DataFrame(records)
    metadata_csv = metadata_dir / "dataset_metadata.csv"
    df.to_csv(metadata_csv, index=False)
    print(f"  Metadata saved to {metadata_csv.relative_to(PROJECT_ROOT)}")

    print("[3/5] Performing deterministic stratified split (Seed=42)...")
    # Total = 3662
    # Required: Train = 2563, Val = 549, Test = 550
    # Step A: Split off Test set (550 samples)
    train_val_df, test_df = train_test_split(
        df,
        test_size=550,
        random_state=42,
        stratify=df["five_class"]
    )
    
    # Step B: Split remaining 3112 into Train (2563) and Val (549)
    # 549 / 3112 = 0.17641388...
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=549,
        random_state=42,
        stratify=train_val_df["five_class"]
    )
    
    assert len(train_df) == 2563, f"Train count mismatch: {len(train_df)} != 2563"
    assert len(val_df) == 549, f"Val count mismatch: {len(val_df)} != 549"
    assert len(test_df) == 550, f"Test count mismatch: {len(test_df)} != 550"
    
    # Check no overlap
    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap detected!"
    print("  Zero split overlap verified across Train, Validation, and Test sets.")

    print("[4/5] Constructing and saving official manifest and indices...")
    train_df = train_df.copy()
    train_df["split"] = "train"
    val_df = val_df.copy()
    val_df["split"] = "val"
    test_df = test_df.copy()
    test_df["split"] = "test"
    
    manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    manifest_path = splits_dir / "official_split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Official manifest saved to {manifest_path.relative_to(PROJECT_ROOT)}")

    indices_data = {
        "train": train_df["image_id"].tolist(),
        "val": val_df["image_id"].tolist(),
        "test": test_df["image_id"].tolist(),
        "counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "total": len(manifest_df)
        },
        "seed": 42
    }
    indices_path = splits_dir / "official_split_indices.json"
    with open(indices_path, "w", encoding="utf-8") as f:
        json.dump(indices_data, f, indent=2)
    print(f"  Split indices saved to {indices_path.relative_to(PROJECT_ROOT)}")

    print("[5/5] Stratified class distribution report:")
    print("-" * 65)
    split_summary = manifest_df.groupby(["split", "five_class"]).size().unstack(fill_value=0)
    split_summary.columns = [f"Class {c} ({CLASS_NAMES[c]})" for c in split_summary.columns]
    split_summary["Total"] = split_summary.sum(axis=1)
    print(split_summary.loc[["train", "val", "test"]])
    print("-" * 65)
    print("Dataset preparation and deterministic stratification completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    prepare_aptos_dataset()
