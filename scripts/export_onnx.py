"""One-time export utility to convert plant_disease_model.h5 to ONNX.

Requires development dependencies (`tensorflow`, `tf2onnx`, `onnx`,
`onnxruntime`). The resulting `plant_disease_model.onnx` runs in production
with only `onnxruntime`, eliminating the >500 MB TensorFlow runtime dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx


def export_h5_to_onnx(
    h5_path: Path,
    onnx_path: Path,
    opset: int = 17,
) -> None:
    """Convert a Keras .h5 CNN model to ONNX and verify numerical parity."""
    if not h5_path.is_file():
        raise FileNotFoundError(f"Source Keras model not found: {h5_path}")

    model = tf.keras.models.load_model(str(h5_path), compile=False)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 128, 128, 3], tf.float32, name="input")
        ]
    )
    def serving_fn(x: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"probabilities": model(x, training=False)}

    tf2onnx.convert.from_function(
        serving_fn,
        input_signature=[
            tf.TensorSpec([None, 128, 128, 3], tf.float32, name="input")
        ],
        opset=opset,
        output_path=str(onnx_path),
    )

    # Verify numerical parity between Keras .h5 and ONNX Runtime on [0, 255] inputs
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(42)

    max_diff = 0.0
    for sample in (
        rng.uniform(0.0, 255.0, size=(1, 128, 128, 3)).astype(np.float32),
        np.full((1, 128, 128, 3), 120.0, dtype=np.float32),
    ):
        tf_probs = model(sample, training=False).numpy()
        ort_probs = session.run(None, {input_name: sample})[0]
        assert int(tf_probs.argmax()) == int(ort_probs.argmax()), (
            "Top-1 class mismatch between Keras and ONNX!"
        )
        max_diff = max(max_diff, float(np.max(np.abs(tf_probs - ort_probs))))

    print(
        f"Exported {h5_path.name} ({h5_path.stat().st_size / 1e6:.2f} MB) -> "
        f"{onnx_path.name} ({onnx_path.stat().st_size / 1e6:.2f} MB) | "
        f"max_abs_diff={max_diff:.2e}"
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "plant_disease_model.h5",
        help="Path to source Keras .h5 model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "plant_disease_model.onnx",
        help="Path to destination ONNX model.",
    )
    args = parser.parse_args()
    export_h5_to_onnx(args.input, args.output)


if __name__ == "__main__":
    main()
