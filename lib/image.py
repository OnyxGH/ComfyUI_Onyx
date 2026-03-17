from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2pil(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    elif image.dim() == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    else:
        image = image.squeeze()

    return Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8))
