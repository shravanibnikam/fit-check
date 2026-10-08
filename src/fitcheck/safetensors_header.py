"""Parse safetensors headers without touching tensor data.

A safetensors file is laid out as:

    8 bytes   N, the header length (unsigned little-endian 64-bit int)
    N bytes   JSON: {"__metadata__": {...}, "<tensor>": {"dtype", "shape", "data_offsets"}}
    the rest  raw tensor bytes

The first 8 + N bytes give every tensor's name, dtype and shape, which is all
we need to count parameters. Format spec:
https://github.com/huggingface/safetensors#format
"""

import json
import math
import struct

PREFIX_BYTES = 8

# The reference implementation rejects anything larger (MAX_HEADER_SIZE in
# safetensors/src/tensor.rs). Guards against reading a corrupt or hostile length.
MAX_HEADER_BYTES = 100_000_000


def header_length(prefix: bytes) -> int:
    """Return N, the JSON header length, from the first 8 bytes of a file."""
    if len(prefix) < PREFIX_BYTES:
        raise ValueError(f"need {PREFIX_BYTES} bytes, got {len(prefix)}")
    (n,) = struct.unpack("<Q", prefix[:PREFIX_BYTES])
    if n > MAX_HEADER_BYTES:
        raise ValueError(f"header length {n} is over {MAX_HEADER_BYTES} bytes")
    return n


def parse_header(raw: bytes) -> dict[str, dict]:
    """Parse the JSON header into {tensor name: {"dtype", "shape", ...}}."""
    if not raw.startswith(b"{"):
        raise ValueError("safetensors header must start with '{'")
    tensors = json.loads(raw)
    tensors.pop("__metadata__", None)
    for name, info in tensors.items():
        shape = info.get("shape") if isinstance(info, dict) else None
        if not (_is_valid_shape(shape) and isinstance(info.get("dtype"), str)):
            # A malformed file, not a caller passing the wrong type.
            raise ValueError(f"tensor {name!r} has no valid dtype and shape")
    return tensors


def _is_valid_shape(shape) -> bool:
    """Dims are non-negative ints whose product fits in a u64.

    The header is untrusted. Without this, math.prod(["x", 10**12]) builds a
    terabyte string, and thousands of 4000-digit dims make a product with
    millions of digits. The reference parser rejects element counts that
    overflow the same way (checked_mul in safetensors/src/tensor.rs).
    """
    if not isinstance(shape, list):
        return False
    n = 1
    for d in shape:
        if not isinstance(d, int) or d < 0:
            return False
        n *= d
        if n >= 2**64:
            return False
    return True


def count_params(tensors: dict[str, dict]) -> int:
    # ponytail: counts stored elements, so packed quantized checkpoints (GPTQ/AWQ
    # int4 packed into I32) undercount; decode the packing when we support them.
    return sum(math.prod(info["shape"]) for info in tensors.values())
