import struct

import pytest

from fitcheck.safetensors_header import (
    MAX_HEADER_BYTES,
    count_params,
    header_length,
    parse_header,
)


def test_round_trip(make_safetensors):
    data = make_safetensors({"embed.weight": [10, 4], "norm.weight": [4], "scale": []})
    n = header_length(data[:8])
    tensors = parse_header(data[8 : 8 + n])
    assert set(tensors) == {"embed.weight", "norm.weight", "scale"}
    assert tensors["embed.weight"]["dtype"] == "BF16"
    assert count_params(tensors) == 10 * 4 + 4 + 1  # a scalar has one element


def test_trailing_space_padding_is_allowed(make_safetensors):
    # Writers may pad the header with spaces to align the data section.
    data = make_safetensors({"w": [2, 3]})
    raw = data[8:] + b"    "
    assert count_params(parse_header(raw)) == 6


def test_rejects_huge_header_length():
    with pytest.raises(ValueError, match="over"):
        header_length(struct.pack("<Q", MAX_HEADER_BYTES + 1))


def test_rejects_short_prefix():
    with pytest.raises(ValueError, match="need 8 bytes"):
        header_length(b"\x00\x01")


def test_rejects_non_json_header():
    with pytest.raises(ValueError, match="start with"):
        parse_header(b"PK\x03\x04 this is a zip")


@pytest.mark.parametrize(
    "entry",
    [
        b'{"dtype": "F16"}',
        b'{"dtype": "F16", "shape": ["x", 1000000000000]}',
        b'{"dtype": "F16", "shape": [-4, 2]}',
        b'{"dtype": "F16", "shape": [4294967296, 4294967296]}',
        b'{"dtype": "F16", "shape": [' + b",".join([b"9" * 4000] * 2000) + b"]}",
        b'{"shape": [2]}',
        b"[1, 2]",
    ],
    ids=[
        "no-shape",
        "string-dim",
        "negative-dim",
        "u64-overflow",
        "huge-dims",
        "no-dtype",
        "not-an-object",
    ],
)
def test_rejects_malformed_tensor_entry(entry):
    with pytest.raises(ValueError, match="'w'"):
        parse_header(b'{"w": ' + entry + b"}")
