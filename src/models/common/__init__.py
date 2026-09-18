"""Common model components shared across modules.

This directory contains reusable components borrowed/adapted from other
projects (e.g., DyRSRNet) to avoid code duplication.
"""

__all__ = ["OSRM"]


def __getattr__(name: str):
    if name == "OSRM":
        from src.models.common.osrm import OSRM
        return OSRM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
