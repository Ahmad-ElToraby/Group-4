# deployment_preprocessing.py
from typing import Tuple

import numpy as np
from PIL import Image


def preprocess_pil_image(
    pil_img: Image.Image,
    img_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    img = pil_img.convert("RGB")
    img = img.resize(img_size)
    img_array = np.array(img).astype("float32")   # 0–255
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array
