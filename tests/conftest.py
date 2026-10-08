import json
import struct

import pytest


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
