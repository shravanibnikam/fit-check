import json
import struct

import pytest

from fitcheck import estimate


@pytest.fixture(autouse=True)
def bare_theory(monkeypatch):
    """Exact-byte tests check the formulas, not whatever calibration.json holds."""
    monkeypatch.setattr(estimate, "RESERVED_OVER_ALLOCATED", 1.0)
    monkeypatch.setattr(estimate, "CUDA_CONTEXT_BYTES", 0)


@pytest.fixture
def make_safetensors():
    """Build the bytes of a safetensors file with headers only, no tensor data."""

    def make(shapes: dict[str, list[int]], dtype: str = "BF16") -> bytes:
        header = {"__metadata__": {"format": "pt"}}
        for name, shape in shapes.items():
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [0, 0]}
        raw = json.dumps(header).encode()
        return struct.pack("<Q", len(raw)) + raw

    return make
