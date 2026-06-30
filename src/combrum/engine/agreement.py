"""Rank-agreement helpers for public distributed controls."""

from __future__ import annotations

import math
import operator
import pickle
from hashlib import sha256
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from combrum.transport.base import Transport

_T = TypeVar("_T")


def _raise_token_error(token: tuple[bool, object, str, str]) -> None:
    if token[2] == "TypeError":
        raise TypeError(token[3])
    raise ValueError(token[3])


def _token_value_label(value: object) -> str:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[1], bytes)
    ):
        return f"(shape={value[0]!r}, nbytes={len(value[1])})"
    return repr(value)


def _agree_token(
    name: str,
    token: tuple[bool, object, str, str],
    transport: Transport,
) -> object:
    if transport.size == 1:
        if not token[0]:
            _raise_token_error(token)
        return token[1]

    with transport.collective():
        root = transport.bcast(token if transport.rank == 0 else None, root=0)
        if token != root:
            local_value = token[1] if token[0] else token[3]
            root_value = root[1] if root[0] else root[3]
            raise ValueError(
                f"{name} must match on every rank;"
                f" rank {transport.rank} has {_token_value_label(local_value)},"
                f" rank 0 has {_token_value_label(root_value)}"
            )
        if not token[0]:
            _raise_token_error(token)
    return token[1]


def _int_token(
    name: str,
    value: object,
    *,
    lower: int | None = None,
) -> tuple[bool, object, str, str]:
    if isinstance(value, (bool, np.bool_)):
        return (
            False,
            0,
            "TypeError",
            f"{name} must be an integer; got bool",
        )
    try:
        out = int(operator.index(value))
    except TypeError:
        return (
            False,
            0,
            "TypeError",
            f"{name} must be an integer; got {type(value).__name__}",
        )
    if lower is not None and out < lower:
        return (
            False,
            out,
            "ValueError",
            f"{name} must be >= {lower}; got {value!r}",
        )
    return (True, out, "", "")


def agree_public_int(
    name: str,
    value: object,
    transport: Transport,
    *,
    lower: int | None = None,
) -> int:
    """Normalize an integer public control and require rank-uniform value."""

    return int(
        _agree_token(name, _int_token(name, value, lower=lower), transport)
    )


def _float_token(
    name: str,
    value: object,
    *,
    lower: float | None = None,
    strict_lower: bool = False,
) -> tuple[bool, object, str, str]:
    if isinstance(value, bool):
        return (
            False,
            0.0,
            "TypeError",
            f"{name} must be a finite float; got bool",
        )
    try:
        out = float(value)
    except (TypeError, ValueError):
        return (
            False,
            0.0,
            "TypeError",
            f"{name} must be a finite float; got {type(value).__name__}",
        )
    if not math.isfinite(out):
        return (
            False,
            out,
            "ValueError",
            f"{name} must be finite; got {value!r}",
        )
    if lower is not None:
        bad = out <= lower if strict_lower else out < lower
        if bad:
            op = ">" if strict_lower else ">="
            return (
                False,
                out,
                "ValueError",
                f"{name} must be {op} {lower:g}; got {value!r}",
            )
    return (True, out, "", "")


def agree_public_float(
    name: str,
    value: object,
    transport: Transport,
    *,
    lower: float | None = None,
    strict_lower: bool = False,
) -> float:
    """Normalize a float public control and require rank-uniform value."""

    return float(
        _agree_token(
            name,
            _float_token(
                name, value, lower=lower, strict_lower=strict_lower
            ),
            transport,
        )
    )


def _bool_token(name: str, value: object) -> tuple[bool, object, str, str]:
    if not isinstance(value, bool):
        return (
            False,
            False,
            "TypeError",
            f"{name} must be a bool; got {type(value).__name__}",
        )
    return (True, bool(value), "", "")


def agree_public_bool(
    name: str, value: object, transport: Transport
) -> bool:
    """Normalize a boolean public control and require rank-uniform value."""

    return bool(_agree_token(name, _bool_token(name, value), transport))


def _choice_token(
    name: str,
    value: object,
    *,
    choices: tuple[Any, ...],
) -> tuple[bool, object, str, str]:
    if value not in choices:
        rendered = ", ".join(repr(choice) for choice in choices)
        return (
            False,
            value,
            "ValueError",
            f"{name} must be one of {rendered}; got {value!r}",
        )
    return (True, value, "", "")


def agree_public_choice(
    name: str,
    value: object,
    transport: Transport,
    *,
    choices: tuple[Any, ...],
) -> object:
    """Require every rank to pass the same value from a finite choice set."""

    return _agree_token(
        name, _choice_token(name, value, choices=choices), transport
    )


def _optional_theta_token(
    name: str,
    value: object,
    *,
    K: int,
) -> tuple[bool, object, str, str]:
    if value is None:
        return (True, None, "", "")
    try:
        theta = getattr(value, "theta_hat")
    except AttributeError:
        return (
            False,
            None,
            "TypeError",
            f"{name} must be None or expose theta_hat;"
            f" got {type(value).__name__}",
        )
    except Exception as exc:
        return (
            False,
            None,
            "ValueError",
            f"{name}.theta_hat could not be read;"
            f" {type(exc).__name__}: {exc}",
        )
    try:
        arr = np.asarray(theta, dtype=np.float64)
    except Exception as exc:
        return (
            False,
            None,
            "TypeError",
            f"{name}.theta_hat must be numeric; {exc}",
        )
    if arr.shape != (K,):
        return (
            False,
            None,
            "ValueError",
            f"{name}.theta_hat must have shape (K,) = ({K},);"
            f" got {arr.shape}",
        )
    if not np.all(np.isfinite(arr)):
        return (
            False,
            None,
            "ValueError",
            f"{name}.theta_hat must be finite",
        )
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return (True, ((K,), arr.tobytes()), "", "")


def agree_public_optional_theta(
    name: str,
    value: object,
    transport: Transport,
    *,
    K: int,
) -> np.ndarray | None:
    """Normalize optional ``theta_hat`` warm starts and require rank agreement."""

    payload = _agree_token(
        name, _optional_theta_token(name, value, K=K), transport
    )
    if payload is None:
        return None
    shape, raw = payload
    theta = np.frombuffer(raw, dtype=np.float64).copy().reshape(shape)
    theta.setflags(write=False)
    return theta


def _pickle_digest_token(
    name: str, value: object
) -> tuple[bool, object, str, str]:
    try:
        payload = pickle.dumps(value, protocol=5)
    except Exception as exc:
        return (
            False,
            None,
            "TypeError",
            f"{name} must be pickle-serializable for distributed agreement;"
            f" {type(exc).__name__}: {exc}",
        )
    return (
        True,
        (
            type(value).__module__,
            type(value).__qualname__,
            sha256(payload).hexdigest(),
        ),
        "",
        "",
    )


def require_public_object_agreement(
    name: str, value: _T, transport: Transport
) -> _T:
    """Require an object-valued public distributed input to match by digest."""

    if transport.size == 1:
        return value
    _agree_token(name, _pickle_digest_token(name, value), transport)
    return value


def reject_multirank_dense_transport(name: str, transport: Transport) -> None:
    """Keep dense public entry points serial-only for the first release."""

    if transport.size != 1:
        raise ValueError(
            f"{name} does not support non-serial transport in combRUM 0.1.0;"
            f" use {name}_distributed for distributed runs"
        )


def collective_call(transport: Transport, fn: Callable[[], _T]) -> _T:
    """Run rank-local work under an agreed failure guard when multirank."""

    if transport.size == 1:
        return fn()
    with transport.collective():
        return fn()


def _floor_token(name: str, value: object) -> tuple[bool, object, str, str]:
    if value is None:
        return (True, None, "", "")
    return _int_token(name, value, lower=0)


def callback_convergence_floor(
    *,
    name: str,
    callback: Callable[[int, Any], int | None] | None,
    iteration: int,
    oracle: Any,
    base_floor: int,
    transport: Transport,
) -> int:
    """Run a public callback safely and apply rank-0's convergence floor."""

    if callback is None:
        return int(base_floor)

    def _run_callback() -> int | None:
        token = _floor_token(name, callback(iteration, oracle))
        if not token[0]:
            _raise_token_error(token)
        value = token[1]
        return None if value is None else int(value)

    if transport.size == 1:
        callback_floor = _run_callback()
    else:
        with transport.collective():
            local_floor = _run_callback()
        callback_floor = transport.bcast(
            local_floor if transport.rank == 0 else None, root=0
        )

    if callback_floor is None:
        return int(base_floor)
    return max(int(base_floor), int(callback_floor))
