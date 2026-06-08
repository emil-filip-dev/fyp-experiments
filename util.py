import sys

import torch


def configure_utf8_output() -> None:
    """
    Force stdout/stderr to UTF-8 so non-ASCII output doesn't crash when stdout is
    redirected to a file/pipe on Windows (default cp1252 can't encode some chars).
    Call once at the start of a script's main().
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def resolve_device(device: str = "gpu") -> torch.device:
    """
    Return the compute device to use for training/inference.
    Pass "gpu"/"cuda" to use the CUDA GPU when available; anything else (or no GPU)
    falls back to CPU.
    """
    if device in ("gpu", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Human-readable device description for log headers."""
    if device.type == "cuda":
        return f"GPU - {torch.cuda.get_device_name(0)}"
    return "CPU"
