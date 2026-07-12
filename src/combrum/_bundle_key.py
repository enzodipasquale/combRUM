"""Bundle-key codec shared by cut rows and step records."""

from __future__ import annotations

import math
import struct

import numpy as np

_MAGIC = b"CB1"
_HEADER = struct.Struct("!3sHH")


def pack_bundle(bundle: np.ndarray) -> bytes:
    """Pack dtype, shape, and raw bytes into an invertible cut key."""

    arr = np.ascontiguousarray(bundle)
    dtype = arr.dtype.str.encode("ascii")
    shape = np.asarray(arr.shape, dtype="<i8")
    return _HEADER.pack(_MAGIC, len(dtype), arr.ndim) + dtype + shape.tobytes() + arr.tobytes()


def pack_bundles(bundles: np.ndarray) -> list[bytes]:
    """Row-wise :func:`pack_bundle` over a 2-D block, one header build.

    Every row shares the header, so a block of n bundles costs one
    ``tobytes`` and n slices instead of n full packs.
    """
    arr = np.ascontiguousarray(bundles)
    if arr.ndim != 2:
        return [pack_bundle(bundle) for bundle in arr]
    dtype = arr.dtype.str.encode("ascii")
    prefix = (
        _HEADER.pack(_MAGIC, len(dtype), 1)
        + dtype
        + np.asarray((arr.shape[1],), dtype="<i8").tobytes()
    )
    row_nbytes = int(arr.shape[1]) * int(arr.dtype.itemsize)
    if row_nbytes == 0:
        return [prefix] * int(arr.shape[0])
    raw = arr.tobytes()
    return [
        prefix + raw[start : start + row_nbytes]
        for start in range(0, len(raw), row_nbytes)
    ]


def canonical_bundle_key(key: bytes) -> bytes:
    """Canonical bytes for explicit bundle keys; opaque keys pass through."""

    if key.startswith(_MAGIC):
        _parse_packed(key)
        return key
    try:
        bundle = unpack_bundle(key)
    except ValueError:
        return key
    return pack_bundle(bundle)


_DTYPES_BY_TAG: dict[bytes, np.dtype] = {}
_SHAPE_STRUCTS: dict[int, struct.Struct] = {}


def _dtype_from_tag(tag: bytes) -> np.dtype:
    dtype = _DTYPES_BY_TAG.get(tag)
    if dtype is None:
        try:
            dtype = np.dtype(tag.decode("ascii"))
        except (UnicodeDecodeError, TypeError) as exc:
            raise ValueError("bundle_key dtype tag is invalid") from exc
        _DTYPES_BY_TAG[tag] = dtype
    return dtype


def _shape_struct(ndim: int) -> struct.Struct:
    fmt = _SHAPE_STRUCTS.get(ndim)
    if fmt is None:
        fmt = struct.Struct(f"<{ndim}q")
        _SHAPE_STRUCTS[ndim] = fmt
    return fmt


def _parse_packed(key: bytes) -> tuple[np.dtype, tuple[int, ...], int]:
    """Validated header of a packed key: (dtype, shape, payload offset).

    The length arithmetic pins the payload exactly, so validation never
    materializes the array.
    """
    if len(key) < _HEADER.size:
        raise ValueError("bundle_key header is truncated")
    _magic, dtype_len, ndim = _HEADER.unpack_from(key)
    dtype_end = _HEADER.size + int(dtype_len)
    shape_end = dtype_end + int(ndim) * 8
    if shape_end > len(key):
        raise ValueError("bundle_key shape metadata is truncated")
    dtype = _dtype_from_tag(key[_HEADER.size : dtype_end])
    shape = _shape_struct(int(ndim)).unpack_from(key, dtype_end)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"negative dimension in bundle_key shape {shape}")
    expected = math.prod(shape) * int(dtype.itemsize)
    if len(key) - shape_end != expected:
        raise ValueError(
            "bundle_key payload has wrong byte length;"
            f" got {len(key) - shape_end}, expected {expected}"
        )
    return dtype, shape, shape_end


def unpack_bundle(key: bytes) -> np.ndarray:
    """Recover a read-only bundle packed by :func:`pack_bundle`."""

    if key.startswith(_MAGIC):
        dtype, shape, payload_start = _parse_packed(key)
        return np.frombuffer(key, dtype=dtype, offset=payload_start).reshape(shape)

    tag, sep, raw = key.partition(b":")
    if sep:
        try:
            dtype = np.dtype(tag.decode("ascii"))
        except (UnicodeDecodeError, TypeError):
            dtype = None
        if dtype is not None and (
            dtype.itemsize == 0 or len(raw) % dtype.itemsize == 0
        ):
            return np.frombuffer(raw, dtype=dtype)
    raise ValueError(
        "bundle_key does not encode an explicit bundle; a generating bundle"
        " is available only for bundle-carrying formulations (e.g. NSlack),"
        " not for aggregate cuts (e.g. OneSlack)"
    )
