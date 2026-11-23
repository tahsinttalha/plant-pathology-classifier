"""Automated unit and API integration tests for the Plant Pathology Classifier."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app  # noqa: E402
from inference import UNCERTAIN_DIAGNOSIS, PlantDiseasePredictor  # noqa: E402


@pytest.fixture(scope="module")
def predictor() -> PlantDiseasePredictor:
    """Load the ONNX Runtime predictor once for unit tests."""
    return PlantDiseasePredictor(
        model_path=PROJECT_ROOT / "plant_disease_model.onnx",
        class_names_path=PROJECT_ROOT / "class_names.json",
        default_threshold=0.75,
    )


@pytest.fixture(scope="module")
def api_client():
    """Create a FastAPI TestClient with lifespan startup/shutdown active."""
    with TestClient(app) as client:
        yield client


def _encode_image(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def test_preprocess_preserves_unscaled_rgb_contract() -> None:
    """Preprocessing must output (1, 128, 128, 3) float32 in [0.0, 255.0]."""
    sample = Image.new("RGB", (256, 192), color=(200, 120, 40))
    tensor = PlantDiseasePredictor.preprocess(sample)

    assert tensor.shape == (1, 128, 128, 3)
    assert tensor.dtype == np.float32
    assert float(tensor.max()) > 1.0
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 255.0


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_predictor_handles_rgb_rgba_and_grayscale(
    predictor: PlantDiseasePredictor, mode: str
) -> None:
    """PlantDiseasePredictor must accept RGB, RGBA, and greyscale PIL images."""
    rng = np.random.default_rng(42)
    if mode == "RGB":
        arr = rng.integers(0, 256, size=(200, 180, 3), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(0, 256, size=(160, 220, 4), dtype=np.uint8)
    else:
        arr = rng.integers(0, 256, size=(150, 150), dtype=np.uint8)

    image = Image.fromarray(arr, mode=mode)
    result = predictor.predict(image)

    assert result["status"] in {"confident", "uncertain"}
    assert isinstance(result["diagnosis"], str) and len(result["diagnosis"]) > 0
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["threshold"] == 0.75
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0.0

    top_preds = result["top_predictions"]
    assert isinstance(top_preds, list) and len(top_preds) == 3
    confidences = [item["confidence"] for item in top_preds]
    assert confidences == sorted(confidences, reverse=True)
    for item in top_preds:
        assert isinstance(item["label"], str)
        assert isinstance(item["confidence"], float)


def test_onnx_parity_with_keras_h5(predictor: PlantDiseasePredictor) -> None:
    """ONNX predictor probabilities must match plant_disease_model.h5 within 1e-5."""
    h5_path = PROJECT_ROOT / "plant_disease_model.h5"
    if not h5_path.is_file():
        pytest.skip("plant_disease_model.h5 not present locally")

    keras_predictor = PlantDiseasePredictor(
        model_path=h5_path,
        class_names_path=PROJECT_ROOT / "class_names.json",
        default_threshold=0.75,
    )
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")

    ort_result = predictor.predict(image)
    tf_result = keras_predictor.predict(image)

    assert ort_result["status"] == tf_result["status"]
    assert ort_result["top_predictions"][0]["label"] == tf_result["top_predictions"][0]["label"]
    assert abs(ort_result["confidence"] - tf_result["confidence"]) <= 1e-4


def test_synthetic_blank_or_noise_image_triggers_uncertain(
    predictor: PlantDiseasePredictor,
) -> None:
    """A synthetic blank/random-noise image should trigger uncertain status under 0.75."""
    rng = np.random.default_rng(2026)
    noise_array = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    noise_image = Image.fromarray(noise_array, mode="RGB")

    result = predictor.predict(noise_image, threshold=0.75)

    assert result["confidence"] < 0.75
    assert result["status"] == "uncertain"
    assert result["diagnosis"] == UNCERTAIN_DIAGNOSIS


def test_api_health_endpoint(api_client: TestClient) -> None:
    """GET /health returns 200 with model status and metadata."""
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "num_classes": 38,
        "threshold": 0.75,
    }


def test_api_root_serves_ui(api_client: TestClient) -> None:
    """GET / serves the redesigned minimalist frontend HTML."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Plant Pathology Classifier" in response.text


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("WEBP", "image/webp"),
    ],
)
def test_api_predict_valid_images(
    api_client: TestClient, fmt: str, mime: str
) -> None:
    """POST /predict returns 200 and structured JSON for JPEG, PNG, and WEBP uploads."""
    sample = Image.new("RGB", (200, 200), color=(48, 132, 62))
    payload = _encode_image(sample, fmt=fmt)

    response = api_client.post(
        "/predict?threshold=0.75",
        files={"file": (f"leaf.{fmt.lower()}", payload, mime)},
    )
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "status",
        "diagnosis",
        "confidence",
        "threshold",
        "latency_ms",
        "top_predictions",
    }
    assert data["threshold"] == 0.75
    assert len(data["top_predictions"]) == 3


def test_api_predict_rejects_non_image_and_corrupt_files(
    api_client: TestClient,
) -> None:
    """POST /predict returns HTTP 400 for non-image MIME types and corrupt image bytes."""
    non_image = api_client.post(
        "/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )
    assert non_image.status_code == 400

    corrupt_image = api_client.post(
        "/predict",
        files={"file": ("broken.jpg", b"\xff\xd8corrupt-bytes", "image/jpeg")},
    )
    assert corrupt_image.status_code == 400

    empty_image = api_client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert empty_image.status_code == 400


def test_api_predict_drive_rejects_non_drive_urls(
    api_client: TestClient,
) -> None:
    """POST /predict/drive returns HTTP 400 for non-Google-Drive URLs."""
    response = api_client.post(
        "/predict/drive",
        json={"url": "https://example.com/not-a-drive-link.jpg"},
    )
    assert response.status_code == 400
