"""Framework-agnostic plant disease inference engine.

Uses ONNX Runtime by default for fast cold starts (~0.2s) and minimal
deployment footprint (Vercel / Docker / Render), while preserving exact
numerical parity with the trained Keras CNN (unscaled [0.0, 255.0] 128x128
RGB float32 tensors).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE: tuple[int, int] = (128, 128)
DEFAULT_THRESHOLD: float = 0.75
UNCERTAIN_DIAGNOSIS: str = "Uncertain — upload a clearer, single-leaf photograph"

_BASE_DIR = Path(__file__).resolve().parent


class PlantDiseasePredictor:
    """Standalone CNN inference wrapper with confidence thresholding."""

    def __init__(
        self,
        model_path: str | Path = _BASE_DIR / "plant_disease_model.onnx",
        class_names_path: str | Path = _BASE_DIR / "class_names.json",
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._model_path = Path(model_path)
        self._class_names_path = Path(class_names_path)

        if not 0.0 <= default_threshold <= 1.0:
            raise ValueError("default_threshold must be between 0.0 and 1.0.")
        self.default_threshold: float = float(default_threshold)

        if not self._class_names_path.is_file():
            raise FileNotFoundError(
                f"Class names file not found: {self._class_names_path}"
            )
        if not self._model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

        with self._class_names_path.open("r", encoding="utf-8") as fh:
            class_names = json.load(fh)

        if not isinstance(class_names, list) or not class_names:
            raise ValueError("class_names.json must contain a non-empty JSON list.")
        self.class_names: list[str] = [str(label) for label in class_names]

        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._keras_model: Any = None

        suffix = self._model_path.suffix.lower()
        if suffix == ".onnx":
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = ort.InferenceSession(
                str(self._model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        elif suffix in {".h5", ".keras"}:
            import tensorflow as tf  # optional local fallback

            self._keras_model = tf.keras.models.load_model(
                str(self._model_path),
                compile=False,
            )
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}'. Expected .onnx, .h5, or .keras."
            )

    @property
    def is_loaded(self) -> bool:
        """Return True when an inference session or model is resident in memory."""
        return self._session is not None or self._keras_model is not None

    @property
    def num_classes(self) -> int:
        """Return the number of supported pathology classes."""
        return len(self.class_names)

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        """Convert a PIL image to a (1, 128, 128, 3) float32 tensor in [0, 255].

        Critical model contract: the CNN was trained via
        `image_dataset_from_directory` on unscaled [0.0, 255.0] RGB tensors.
        Pixel values MUST NOT be divided by 255.0.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL.Image.Image instance.")

        rgb_image = image.convert("RGB")
        resized = rgb_image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        tensor = np.asarray(resized, dtype=np.float32)
        return np.expand_dims(tensor, axis=0)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        if self._session is not None and self._input_name is not None:
            outputs = self._session.run(None, {self._input_name: batch})
            return np.asarray(outputs[0][0], dtype=np.float32)
        if self._keras_model is not None:
            raw_output = self._keras_model(batch, training=False)
            return np.asarray(raw_output[0], dtype=np.float32)
        raise RuntimeError("Inference engine is not initialized.")

    def predict(
        self,
        image: Image.Image,
        threshold: float | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Run inference on a PIL image and apply confidence thresholding.

        Args:
            image: Input PIL image in any colour mode (RGB, RGBA, L, etc.).
            threshold: Optional override for the minimum acceptance probability
                in [0.0, 1.0]. Defaults to `self.default_threshold` (0.75).
            top_k: Number of ranked candidate classes to include (default 3).

        Returns:
            Structured prediction payload with status, diagnosis, confidence,
            threshold, latency_ms, and top_predictions.
        """
        active_threshold = (
            self.default_threshold if threshold is None else float(threshold)
        )
        if not 0.0 <= active_threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        k = max(1, min(int(top_k), self.num_classes))

        start = time.perf_counter()
        batch = self.preprocess(image)
        probabilities = self._forward(batch)
        latency_ms = (time.perf_counter() - start) * 1000.0

        top_indices = np.argsort(probabilities)[-k:][::-1]
        top_predictions = [
            {
                "label": self.class_names[int(idx)],
                "confidence": round(float(probabilities[int(idx)]), 4),
            }
            for idx in top_indices
        ]

        top_label = top_predictions[0]["label"]
        top_probability = float(probabilities[int(top_indices[0])])
        rounded_confidence = round(top_probability, 4)

        if top_probability < active_threshold:
            status = "uncertain"
            diagnosis = UNCERTAIN_DIAGNOSIS
        else:
            status = "confident"
            diagnosis = top_label

        return {
            "status": status,
            "diagnosis": diagnosis,
            "confidence": rounded_confidence,
            "threshold": round(active_threshold, 4),
            "latency_ms": round(latency_ms, 2),
            "top_predictions": top_predictions,
        }
