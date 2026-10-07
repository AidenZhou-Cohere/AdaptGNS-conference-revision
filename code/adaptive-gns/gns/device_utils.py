"""Explicit device selection and graph-execution provenance.

Metal runs neural kernels on MPS and uses the explicitly named scipy_host
backend for neighbor construction on CPU. No unsupported-kernel fallback is
enabled by this module.
"""
import os
import torch


def as_device(device):
    """DDP uses integer CUDA ranks; serial callers use names/devices."""
    return torch.device("cuda", device) if isinstance(device, int) else torch.device(device)


def resolve_device(requested="auto"):
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback was used")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this process; "
                               "Metal may require access outside a restrictive sandbox")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
            raise RuntimeError("MPS requires PYTORCH_ENABLE_MPS_FALLBACK=0 before "
                               "starting Python; silent CPU kernel fallback is disabled")
    return torch.device(requested)


def resolve_radius_backend(requested, device):
    device = as_device(device)
    if requested is None or requested == "auto":
        return "scipy_host" if device.type == "mps" else "pyg"
    if requested not in {"pyg", "scipy", "scipy_host"}:
        raise ValueError("radius_backend must be auto, pyg, scipy, or scipy_host")
    if requested == "pyg" and device.type == "mps":
        raise ValueError("PyG radius_graph has no native MPS backend; explicitly use "
                         "radius_backend=scipy_host (CPU neighbor search and tensor transfer)")
    if requested == "scipy" and device.type != "cpu":
        raise ValueError("scipy is CPU-only; use scipy_host for explicit CPU graph "
                         "construction with accelerator tensor transfer")
    return requested


def runtime_provenance(device, radius_backend):
    return {
        "device": str(as_device(device)),
        "radius_backend": radius_backend,
        "graph_execution": ("cpu_scipy_with_device_transfer" if radius_backend == "scipy_host"
                            else "cpu_scipy" if radius_backend == "scipy" else "pyg"),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "torch_version": str(torch.__version__),
    }


def synchronize(device):
    device = as_device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def add_device_argument(parser):
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
                        help="auto prefers CUDA, then Metal, then CPU; explicit unavailable devices fail")
