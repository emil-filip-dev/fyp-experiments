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


def resolve_device(device: str = "auto") -> torch.device:
    """
    Return the compute device to use for training/inference.
      "cpu"                 -> always CPU
      "auto"/"gpu"/"cuda"   -> CUDA GPU when available, else CPU
    Default "auto" so a GPU box is used without having to ask for it.
    """
    if str(device).lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Human-readable device description for log headers."""
    if device.type == "cuda":
        return f"GPU - {torch.cuda.get_device_name(0)}"
    return "CPU"
