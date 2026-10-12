"""FashionCLIP: a CLIP ViT-B/32 model fine-tuned on fashion product images and text."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

MODEL_NAME = "patrickjohncyh/fashion-clip"


class Encoder(Protocol):
    name: str
    dim: int

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: list[str]) -> np.ndarray: ...


def _projected(features) -> np.ndarray:
    # Transformers 5 returns an output object whose pooler_output holds the
    # projected embedding; earlier versions return the tensor itself.
    tensor = getattr(features, "pooler_output", features)
    return tensor.float().cpu().numpy()


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FashionClipEncoder:
    """Unit-length embeddings from the FashionCLIP image and text towers."""

    def __init__(self, device: str | None = None, model_name: str = MODEL_NAME):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            # Use the local cache when the model is already downloaded; this
            # skips the Hugging Face Hub update check and starts much faster.
            model = CLIPModel.from_pretrained(
                model_name, torch_dtype=dtype, local_files_only=True
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            model = CLIPModel.from_pretrained(model_name, torch_dtype=dtype)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = model.to(self.device).eval()
        self.name = model_name
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.dim = int(self.model.config.projection_dim)
        self.dtype = dtype

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device, self.dtype)
        with self.torch.inference_mode():
            features = self.model.get_image_features(pixel_values=pixels)
        return normalize(_projected(features))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return normalize(_projected(features))
