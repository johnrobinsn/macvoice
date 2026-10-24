"""Memory measurement that counts GPU memory on Apple Silicon.

``ru_maxrss`` and Activity Monitor's "Memory" miss most Metal allocations
on unified memory. The kernel's *physical footprint* counts them, so it is
the number that compares fairly across PyTorch-MPS, MLX and Rust/Metal
runtimes in one process.
"""

from __future__ import annotations

import ctypes
import os
import sys


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(36)]


# rusage_info_v4 fields after the uuid (sys/resource.h)
_PHYS_FOOTPRINT = 7
_LIFETIME_MAX_PHYS_FOOTPRINT = 28


def _rusage() -> _RUsageInfoV4 | None:
    if sys.platform != "darwin":
        return None
    info = _RUsageInfoV4()
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return info


def footprint_mb() -> float | None:
    """Current physical footprint in MB (None off macOS)."""
    info = _rusage()
    return None if info is None else getattr(info, f"f{_PHYS_FOOTPRINT}") / 2**20


def peak_footprint_mb() -> float | None:
    """Lifetime-maximum physical footprint in MB (None off macOS)."""
    info = _rusage()
    return None if info is None else getattr(info, f"f{_LIFETIME_MAX_PHYS_FOOTPRINT}") / 2**20


def torch_loaded() -> bool:
    """Whether anything in this process has imported PyTorch."""
    return "torch" in sys.modules
