from typing import Optional

import torch

_PIN_MEMORY: Optional[bool] = None


def configure_pin_memory(
    accelerator: Optional[str],
    cuda_available: bool,
) -> None:
    global _PIN_MEMORY
    _PIN_MEMORY = bool(cuda_available and accelerator not in {"cpu", "mps"})


def should_pin_memory() -> bool:
    if _PIN_MEMORY is not None:
        return _PIN_MEMORY
    return torch.cuda.is_available()
