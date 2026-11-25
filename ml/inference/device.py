"""Accelerator detection: CUDA > MPS > CPU, with explicit MPS fallback handling."""
from __future__ import annotations
import os
import torch

def detect_device(prefer: str | None = None) -> torch.device:
    prefer = (prefer or os.getenv("POLYVOICE_DEVICE") or "auto").lower()
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # Let unsupported ops fall back to CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")

def memory_allocated_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.current_allocated_memory() / 2**20
    return None
