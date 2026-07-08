"""Resource probes: wall clock and peak RSS.

In-process ``measure`` plus ``ru_maxrss`` unit normalization.
Measurement only; nothing here decides whether a number is acceptable.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

_T = TypeVar("_T")


def normalize_maxrss(ru_maxrss: int, platform: str) -> int:
    """Normalize a ``getrusage`` ``ru_maxrss`` reading to bytes.

    darwin reports bytes, linux reports kilobytes (scaled by 1024).
    Unknown platforms are rejected rather than guessing a unit.
    """
    if platform == "darwin":
        return int(ru_maxrss)
    if platform.startswith("linux"):
        return int(ru_maxrss) * 1024
    raise ValueError(
        f"ru_maxrss unit convention unknown for platform {platform!r}"
    )


@dataclass(frozen=True)
class ProbeReport:
    """One measurement: wall-clock seconds and peak RSS in bytes.

    Both fields must be strictly positive; a non-positive reading is a
    broken probe.
    """

    wall_seconds: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.wall_seconds, bool) or not isinstance(
            self.wall_seconds, (int, float)
        ):
            raise ValueError(
                f"wall_seconds must be a float > 0; got {self.wall_seconds!r}"
            )
        wall = float(self.wall_seconds)
        if not math.isfinite(wall) or wall <= 0.0:
            raise ValueError(
                f"wall_seconds must be a finite float > 0; got {wall!r}"
            )
        object.__setattr__(self, "wall_seconds", wall)
        if (
            isinstance(self.peak_rss_bytes, bool)
            or not isinstance(self.peak_rss_bytes, int)
            or self.peak_rss_bytes <= 0
        ):
            raise ValueError(
                "peak_rss_bytes must be an integer > 0;"
                f" got {self.peak_rss_bytes!r}"
            )


def measure(fn: Callable[[], _T]) -> tuple[_T, ProbeReport]:
    """Run ``fn`` in-process and report wall clock plus peak RSS.

    ``ru_maxrss`` is a process-lifetime high-water mark, so the reported
    peak is the lifetime peak at call end, not the call's own allocation.
    """
    start = time.perf_counter()
    result = fn()
    wall = time.perf_counter() - start
    peak = normalize_maxrss(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    return result, ProbeReport(wall_seconds=wall, peak_rss_bytes=peak)


# ---------------------------------------------------------------------------
# Tests for this support module. Collected only when this file is named
# explicitly on the pytest command line (the default ``python_files =
# test_*.py`` glob skips it).
#
# Expected values never route through ``measure``'s own arithmetic: the wall
# reading is checked against an exact hand-chosen perf_counter delta, the
# peak reading against an independently sampled ru_maxrss from a fresh
# subprocess (so suite run order cannot affect it).
# ---------------------------------------------------------------------------

# A 256 MiB buffer dwarfs a fresh interpreter's resident set, so touching every
# page raises the lifetime ru_maxrss mark well past the child's own baseline.
_PROBE_ALLOC_BYTES = 256 * 1024 * 1024
# Headroom below the allocation: a real rising peak clears it; a peak that does
# not reflect the buffer (sampled before fn, fabricated, or dropped) cannot.
_PROBE_RISE_FLOOR = 64 * 1024 * 1024


def _probe_touch_pages(buf: "bytearray") -> None:
    for offset in range(0, len(buf), 4096):
        buf[offset] = 1


def _probe_peak_child(conn: object, variant: str) -> None:
    """Run ``measure`` on a page-touching buffer inside a fresh interpreter.

    Executed as a ``multiprocessing`` (spawn) target so its ru_maxrss
    high-water mark starts from a clean baseline, unaffected by what the
    parent test process already allocated. ``variant`` swaps in a broken
    ``measure`` ("prefn" samples the peak before fn, "overscale" hardcodes
    the linux unit); "" runs the real one.
    """
    import resource as _resource
    import sys as _sys
    import time as _time

    baseline = normalize_maxrss(
        _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss, _sys.platform
    )
    keep: dict[str, bytearray] = {}

    def work() -> int:
        buf = bytearray(_PROBE_ALLOC_BYTES)
        _probe_touch_pages(buf)
        keep["buf"] = buf  # hold the buffer so the mark still reflects it
        return sum(range(1000))

    if variant == "prefn":
        # measure samples the peak before running fn -> the buffer never lands.
        def _measure_prefn(
            fn: Callable[[], _T],
        ) -> tuple[_T, ProbeReport]:
            start = _time.perf_counter()
            peak = normalize_maxrss(
                _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss,
                _sys.platform,
            )
            res = fn()
            wall = _time.perf_counter() - start
            return res, ProbeReport(wall_seconds=wall, peak_rss_bytes=peak)

        result, report = _measure_prefn(work)
    elif variant == "overscale":
        # measure hardcodes the linux (kib) unit -> 1024x over-report on darwin.
        def _measure_overscale(
            fn: Callable[[], _T],
        ) -> tuple[_T, ProbeReport]:
            start = _time.perf_counter()
            res = fn()
            wall = _time.perf_counter() - start
            raw = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            peak = int(raw) * 1024
            return res, ProbeReport(wall_seconds=wall, peak_rss_bytes=peak)

        result, report = _measure_overscale(work)
    else:
        result, report = measure(work)

    # Re-sample the mark (buffer still held) and normalize by hand,
    # independently of measure's output.
    post_bytes = normalize_maxrss(
        _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss, _sys.platform
    )
    conn.send(
        (result, int(report.peak_rss_bytes), int(post_bytes), int(baseline))
    )
    conn.close()


def _run_probe_peak_child(variant: str = "") -> tuple[object, int, int, int]:
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_probe_peak_child, args=(child_conn, variant))
    proc.start()
    payload = parent_conn.recv()
    proc.join(timeout=60)
    if proc.exitcode != 0:
        raise AssertionError(f"probe child exited with {proc.exitcode!r}")
    return payload


class _StepClock:
    """Deterministic perf_counter stand-in yielding a fixed value sequence."""

    def __init__(self, values: "tuple[float, ...]") -> None:
        self._values = list(values)
        self._index = 0

    def __call__(self) -> float:
        value = self._values[self._index]
        self._index += 1
        return value


def test_measure_wall_equals_exact_perf_counter_delta(monkeypatch) -> None:
    # Endpoints hand-chosen so the difference is float-exact
    # (1.5 - 1.25 == 0.25); exact equality rejects any additive or
    # multiplicative distortion of the reading.
    start_tick = 1.25
    end_tick = 1.5
    expected_wall = end_tick - start_tick  # 0.25, exactly representable

    monkeypatch.setattr(
        time, "perf_counter", _StepClock((start_tick, end_tick))
    )
    _, report = measure(lambda: None)
    assert report.wall_seconds == expected_wall


def test_measure_wall_tracks_a_distorted_clock(monkeypatch) -> None:
    # A clock that inflates the interval by 25% (end = start + 1.25 * delta)
    # moves measure's reading to 0.3125, away from the true 0.25 — the
    # exact-delta comparison above is not satisfiable by accident.
    start_tick = 1.25
    true_delta = 0.25
    inflated_end = start_tick + 1.25 * true_delta  # 1.5625, exact
    monkeypatch.setattr(
        time, "perf_counter", _StepClock((start_tick, inflated_end))
    )
    _, report = measure(lambda: None)
    assert report.wall_seconds != true_delta


def test_measure_peak_uses_real_platform_unit_in_process() -> None:
    # fn allocates nothing, so the mark is stable across the call: measure's
    # peak must land between independent samples taken immediately before and
    # after, both normalized by hand. Hardcoding the linux unit (1024x on
    # darwin) would land far above the upper bracket.
    pristine_normalize = normalize_maxrss
    before = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    _, report = measure(lambda: None)
    after = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    assert before <= report.peak_rss_bytes <= after


def test_measure_peak_overscale_falls_outside_bracket(
    monkeypatch,
) -> None:
    # Patch the name measure resolves so it over-scales, while the bracket
    # keeps a pristine reference: the reading must leave the bracket.
    pristine_normalize = normalize_maxrss

    def _linux_unit(ru_maxrss: int, platform: str) -> int:
        return int(ru_maxrss) * 1024

    before = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    monkeypatch.setattr(sys.modules[__name__], "normalize_maxrss", _linux_unit)
    _, report = measure(lambda: None)
    after = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    assert not (before <= report.peak_rss_bytes <= after)


def test_measure_peak_equals_independent_maxrss_reading() -> None:
    # A fresh subprocess re-reads its own ru_maxrss (buffer held alive) and
    # normalizes by hand, so the expected value never routes through measure
    # and a prior memory-heavy test in the parent cannot skew it.
    result, peak, post_bytes, baseline = _run_probe_peak_child()
    assert result == 499500
    # Exact equality: unit errors, pre-call sampling, and fabricated
    # constants all fail the same comparison.
    assert peak == post_bytes
    # The buffer must have raised the fresh child's mark well past baseline.
    assert peak >= baseline + _PROBE_RISE_FLOOR


def test_measure_peak_sampled_before_fn_misses_the_rise() -> None:
    # Under-report direction: sampling the peak before fn leaves the child's
    # rise at zero, below the floor.
    _, peak, _post, baseline = _run_probe_peak_child(variant="prefn")
    assert peak < baseline + _PROBE_RISE_FLOOR


def test_measure_peak_overscaled_unit_diverges_from_reread() -> None:
    # Over-report direction: hardcoding the linux kib unit reports 1024x the
    # real darwin bytes, so the reading no longer matches the re-read mark.
    _, peak, post_bytes, _baseline = _run_probe_peak_child(variant="overscale")
    assert peak != post_bytes


# ---------------------------------------------------------------------------
# normalize_maxrss branches, with explicit platform strings so each branch
# runs regardless of the host. Expected values apply the stated unit
# convention by hand: darwin bytes pass through, linux kib scale by 1024.
# ---------------------------------------------------------------------------


def test_normalize_maxrss_darwin_identity() -> None:
    assert normalize_maxrss(123456, "darwin") == 123456


def test_normalize_maxrss_linux_scales_by_1024() -> None:
    assert normalize_maxrss(123456, "linux") == 123456 * 1024
    assert normalize_maxrss(1, "linux") == 1024


def test_normalize_maxrss_linux_prefix_family_scaled() -> None:
    # The kib->byte contract keys on the "linux" prefix, not the exact
    # literal, so "linux2" scales rather than raising.
    assert normalize_maxrss(7, "linux2") == 7 * 1024


def test_normalize_maxrss_unknown_platform_rejected() -> None:
    # Unknown platforms raise instead of guessing a unit.
    import pytest

    with pytest.raises(ValueError, match="win32"):
        normalize_maxrss(1, "win32")


# ---------------------------------------------------------------------------
# ProbeReport.__post_init__. measure() only constructs valid reports, so the
# rejected boundaries need direct construction.
# ---------------------------------------------------------------------------


def test_probe_report_rejects_broken_readings() -> None:
    import pytest

    for kwargs in (
        dict(wall_seconds=0.0, peak_rss_bytes=1),
        dict(wall_seconds=-1.0, peak_rss_bytes=1),
        dict(wall_seconds=float("nan"), peak_rss_bytes=1),
        dict(wall_seconds=float("inf"), peak_rss_bytes=1),
        dict(wall_seconds=True, peak_rss_bytes=1),
        dict(wall_seconds="0.5", peak_rss_bytes=1),
        dict(wall_seconds=0.1, peak_rss_bytes=0),
        dict(wall_seconds=0.1, peak_rss_bytes=-5),
        dict(wall_seconds=0.1, peak_rss_bytes=2.5),
        dict(wall_seconds=0.1, peak_rss_bytes=True),
    ):
        with pytest.raises(ValueError):
            ProbeReport(**kwargs)  # type: ignore[arg-type]


def test_probe_report_coerces_int_wall_to_float() -> None:
    report = ProbeReport(wall_seconds=2, peak_rss_bytes=1)
    assert type(report.wall_seconds) is float
    assert report.wall_seconds == 2.0
