import torch
import csv
import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.dataset import RetinalFundusDataset
from src.inference.engine import DRInferenceEngine
from src.models.network import DRMultiDomainModel
from src.processing.validation import validate_fundus_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = PROJECT_ROOT / "data" / "raw" / "images" / "aptos_2389.png"


@pytest.fixture(scope="module")
def inference_engine():
    return DRInferenceEngine(device="cpu")


def test_dataset_sample_shape():
    dataset = RetinalFundusDataset(split="train", mode="proposed_arps")
    sample = dataset[0]

    assert sample["spatial"].shape == (3, 224, 224)
    assert sample["vessel"].shape == (1, 224, 224)
    assert sample["frequency"].shape == (1, 224, 224)
    assert sample["binary_label"].shape == torch.Size([])
    assert sample["five_class_label"].shape == torch.Size([])


def test_single_sample_forward_pass():
    dataset = RetinalFundusDataset(split="train", mode="proposed_arps")
    sample = dataset[0]
    model = DRMultiDomainModel(num_classes=2, include_vessel=True, include_frequency=True)

    logits = model(
        sample["spatial"].unsqueeze(0),
        sample["vessel"].unsqueeze(0),
        sample["frequency"].unsqueeze(0),
    )

    assert logits.shape == (1, 2)


def test_make_quality_quartiles_handles_duplicate_edges():
    import pandas as pd

    from scripts.run_robustness import make_quality_quartiles

    values = pd.Series([0.0, 0.0, 2.9652, 2.9652, 4.4478])
    quartiles = make_quality_quartiles(values)

    assert list(quartiles.unique()) == ["Q1", "Q2", "Q3", "Q4"]


def test_inference_artifacts_save_prediction_and_gradcam(tmp_path, monkeypatch):
    import app.app as streamlit_app

    monkeypatch.setattr(streamlit_app, "PROJECT_ROOT", tmp_path)
    result = SimpleNamespace(
        gradcam_overlay_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        five_class_gradcam_overlay_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        is_valid=True,
        selected_path="A",
        binary_prediction_str="Normal",
        binary_probabilities={"Normal": 99.0, "DR": 1.0},
        five_class_prediction_str="No DR",
        five_class_probabilities={"No DR": 99.0},
        confidence=0.99,
        consistency={"consistency_status": "MATCH"},
        error_message=None,
    )

    streamlit_app.save_inference_artifacts(result, "test-digest")

    log_path = tmp_path / "logs" / "inference" / "predictions.csv"
    with log_path.open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    assert len(rows) == 1
    assert rows[0]["image_digest"] == "test-digest"
    assert rows[0]["binary_prediction"] == "Normal"
    assert (tmp_path / "figures" / "gradcam" / "test-digest_binary.png").is_file()
    assert (tmp_path / "figures" / "gradcam" / "test-digest_five_class.png").is_file()


def test_official_split_integrity():
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "official_split_manifest.csv")
    indices = json.loads((PROJECT_ROOT / "data" / "splits" / "official_split_indices.json").read_text(encoding="utf-8"))
    expected_counts = {"train": 2563, "val": 549, "test": 550}

    assert len(manifest) == 3662
    assert manifest["image_id"].is_unique
    assert manifest.groupby("split").size().to_dict() == expected_counts
    for split, expected_count in expected_counts.items():
        manifest_ids = set(manifest.loc[manifest["split"] == split, "image_id"])
        index_ids = set(indices[split])
        assert len(index_ids) == expected_count
        assert index_ids == manifest_ids
    assert not (set(indices["train"]) & set(indices["val"]))
    assert not (set(indices["train"]) & set(indices["test"]))
    assert not (set(indices["val"]) & set(indices["test"]))


def test_image_validation_accepts_fundus_and_rejects_non_image(tmp_path):
    valid_result = validate_fundus_image(TEST_IMAGE)
    assert valid_result.is_valid
    assert valid_result.image_bgr is not None

    non_image = tmp_path / "not_an_image.txt"
    non_image.write_text("not an image", encoding="utf-8")
    invalid_result = validate_fundus_image(non_image)
    assert not invalid_result.is_valid
    assert "Unsupported format" in invalid_result.error_message

    corrupt_image = tmp_path / "corrupt.png"
    corrupt_image.write_bytes(b"not a decodable image")
    corrupt_result = validate_fundus_image(corrupt_image)
    assert not corrupt_result.is_valid
    assert "Failed to read image" in corrupt_result.error_message


def test_missing_checkpoints_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="refusing random-weight inference"):
        DRInferenceEngine(
            binary_model_path=tmp_path / "missing_binary.pt",
            five_class_model_path=tmp_path / "missing_five_class.pt",
            device="cpu",
        )


def test_cpu_single_and_batch_inference_with_gradcam(inference_engine):
    assert next(inference_engine.binary_model.parameters()).device.type == "cpu"
    assert next(inference_engine.five_class_model.parameters()).device.type == "cpu"

    dataset = RetinalFundusDataset(split="test", mode="raw")
    second_image = PROJECT_ROOT / dataset.df.iloc[1]["file_path"]
    results = inference_engine.predict_batch([TEST_IMAGE, second_image])

    assert len(results) == 2
    for result in results:
        assert result.is_valid
        assert result.binary_prediction_str in {"Normal", "DR"}
        assert result.five_class_label in range(5)
        assert sum(result.binary_probabilities.values()) == pytest.approx(100.0, abs=0.03)
        assert sum(result.five_class_probabilities.values()) == pytest.approx(100.0, abs=0.05)
        assert result.consistency["consistency_status"] in {"MATCH", "WARNING"}
        assert result.gradcam_overlay_rgb is not None
        assert result.five_class_gradcam_overlay_rgb is not None
        assert result.arps_details is not None
        assert set(result.arps_details.candidates) == {"PATH_A", "PATH_B", "PATH_C"}
