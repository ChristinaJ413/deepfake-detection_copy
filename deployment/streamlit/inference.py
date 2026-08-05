from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


class DeepfakeDetector:
    """Load an ONNX ResNet-50 model and classify face images."""

    def __init__(
        self,
        model_path: str | Path,
        decision_threshold: float = 0.45,
        output_is_logit: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.decision_threshold = decision_threshold
        self.output_is_logit = output_is_logit

        if not self.model_path.exists():
            raise FileNotFoundError(
                "ONNX model not found at:\n"
                f"{self.model_path.resolve()}\n\n"
                "Confirm that both resnet50_model.onnx and "
                "resnet50_model.onnx.data are in the models folder."
            )

        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError(
                "Decision threshold must be between 0 and 1."
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        self.input_name = input_info.name
        self.output_name = output_info.name
        self.input_shape = input_info.shape

        if len(self.input_shape) != 4:
            raise ValueError(
                "Expected a four-dimensional image input, "
                f"but the model reports {self.input_shape}."
            )

        self.layout = self._detect_layout(self.input_shape)

        if self.layout == "NCHW":
            self.image_height = self._dimension_or_default(
                self.input_shape[2], 224
            )
            self.image_width = self._dimension_or_default(
                self.input_shape[3], 224
            )
        else:
            self.image_height = self._dimension_or_default(
                self.input_shape[1], 224
            )
            self.image_width = self._dimension_or_default(
                self.input_shape[2], 224
            )

    @staticmethod
    def _dimension_or_default(
        value: object,
        default: int,
    ) -> int:
        if isinstance(value, int) and value > 0:
            return value
        return default

    @staticmethod
    def _detect_layout(shape: list[object]) -> str:
        if shape[1] == 3:
            return "NCHW"

        if shape[-1] == 3:
            return "NHWC"

        raise ValueError(
            "Could not determine whether the model uses NCHW or NHWC. "
            f"Model input shape: {shape}"
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Calculate sigmoid safely for large positive or negative logits."""
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))

        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Match the evaluation preprocessing used by ResNet-50."""

        if image is None:
            raise ValueError("No image was provided.")

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = image.resize(
            (self.image_width, self.image_height),
            Image.Resampling.BILINEAR,
        )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - IMAGENET_MEAN) / IMAGENET_STD

        if self.layout == "NCHW":
            array = np.transpose(array, (2, 0, 1))

        array = np.expand_dims(array, axis=0)

        return array.astype(np.float32, copy=False)

    def predict(
        self,
        image: Image.Image,
    ) -> dict[str, float | str]:
        """Return the predicted class and class probabilities."""

        input_tensor = self.preprocess(image)

        raw_output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]

        output_values = np.asarray(raw_output).reshape(-1)

        if output_values.size != 1:
            raise RuntimeError(
                "Expected one binary model output, but received "
                f"shape {np.asarray(raw_output).shape}."
            )

        raw_score = float(output_values[0])

        if self.output_is_logit:
            fake_probability = self._sigmoid(raw_score)
        else:
            fake_probability = float(
                np.clip(raw_score, 0.0, 1.0)
            )

        real_probability = 1.0 - fake_probability

        prediction = (
            "Fake"
            if fake_probability >= self.decision_threshold
            else "Real"
        )

        return {
            "prediction": prediction,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "raw_score": raw_score,
            "decision_threshold": self.decision_threshold,
        }
