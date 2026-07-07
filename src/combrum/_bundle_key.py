"""Bundle-key codec shared by cut rows and step records."""

from __future__ import annotations

import struct

import numpy as np

_MAGIC = b"CB1"
_HEADER = struct.Struct("!3sHH")


def pack_bundle(bundle: np.ndarray) -> bytes:
    """Pack dtype, shape, and raw bytes into an invertible cut key."""

    arr = np.ascontiguousarray(bundle)
    dtype = arr.dtype.str.encode("ascii")
    if len(dtype) > np.iinfo(np.uint16).max:
        raise ValueError(f"bundle dtype tag is too long: {arr.dtype.str!r}")
    if arr.ndim > np.iinfo(np.uint16).max:
        raise ValueError(f"bundle has too many dimensions: {arr.ndim}")
    shape = np.asarray(arr.shape, dtype="<i8")
    return _HEADER.pack(_MAGIC, len(dtype), arr.ndim) + dtype + shape.tobytes() + arr.tobytes()


def canonical_bundle_key(key: bytes) -> bytes:
    """Canonical bytes for explicit bundle keys; opaque keys pass through."""

    if key.startswith(_MAGIC):
        unpack_bundle(key)
        return key
    try:
        bundle = unpack_bundle(key)
    except ValueError:
        return key
    return pack_bundle(bundle)


def unpack_bundle(key: bytes) -> np.ndarray:
    """Recover a read-only bundle packed by :func:`pack_bundle`."""

    if key.startswith(_MAGIC):
        if len(key) < _HEADER.size:
            raise ValueError("bundle_key header is truncated")
        _magic, dtype_len, ndim = _HEADER.unpack_from(key)
        pos = _HEADER.size
        dtype_end = pos + int(dtype_len)
        shape_end = dtype_end + int(ndim) * 8
        if shape_end > len(key):
            raise ValueError("bundle_key shape metadata is truncated")
        try:
            dtype = np.dtype(key[pos:dtype_end].decode("ascii"))
        except (UnicodeDecodeError, TypeError) as exc:
            raise ValueError("bundle_key dtype tag is invalid") from exc
        shape_arr = np.frombuffer(key, dtype="<i8", count=int(ndim), offset=dtype_end)
        shape = tuple(int(v) for v in shape_arr)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"bundle_key shape must be nonnegative; got {shape}")
        raw = key[shape_end:]
        expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if len(raw) != expected:
            raise ValueError(
                "bundle_key payload has wrong byte length;"
                f" got {len(raw)}, expected {expected}"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

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
