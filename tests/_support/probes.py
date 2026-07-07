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
# Tests for this support module.
#
# Collected only when this file is named explicitly on the pytest command line
# (the default ``python_files = test_*.py`` glob skips it), which is how the
# probe re-audit exercises them. They pin ``measure``'s two readings against
# oracles that do not route through ``measure``'s own arithmetic:
#   * the wall reading is pinned to an exact hand-chosen perf_counter delta,
#   * the peak reading is pinned to an independently sampled ru_maxrss taken in
#     a fresh subprocess, so it does not depend on suite run order.
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


def _probe_peak_child(conn: object, mutate: str) -> None:
    """Run ``measure`` on a page-touching buffer inside a fresh interpreter.

    Executed as a ``multiprocessing`` (spawn) target so its ru_maxrss high-water
    mark starts from a clean baseline, unaffected by whatever the parent test
    process already allocated. ``mutate`` selects an in-child corruption of the
    peak reading used by the bidirectional regression proof; "" runs the real
    ``measure`` unchanged.
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

    if mutate == "prefn":
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
    elif mutate == "overscale":
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

    # Independent post reading: re-sample the mark ourselves (buffer still held)
    # and normalize by hand. Nothing here is copied from measure's own output.
    post_bytes = normalize_maxrss(
        _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss, _sys.platform
    )
    conn.send(
        (result, int(report.peak_rss_bytes), int(post_bytes), int(baseline))
    )
    conn.close()


def _run_probe_peak_child(mutate: str = "") -> tuple[object, int, int, int]:
    import multiprocessing as _mp

    ctx = _mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    proc = ctx.Process(target=_probe_peak_child, args=(child_conn, mutate))
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
    # Control perf_counter so the elapsed interval is exactly known, then pin
    # measure's wall to that interval by equality. The endpoints are hand
    # chosen so their difference is float-exact (1.5 - 1.25 == 0.25); the
    # expected wall is that subtraction (definition of elapsed time), not a
    # value read back from measure. Exact equality rejects every additive and
    # multiplicative distortion at once, not just one factor.
    start_tick = 1.25
    end_tick = 1.5
    expected_wall = end_tick - start_tick  # 0.25, exactly representable

    monkeypatch.setattr(
        time, "perf_counter", _StepClock((start_tick, end_tick))
    )
    _, report = measure(lambda: None)
    assert report.wall_seconds == expected_wall


def test_measure_wall_oracle_catches_over_report(monkeypatch) -> None:
    # Bidirectional proof for the wall pin: a timer that over-reports the
    # interval by 25% (end = start + 1.25 * true_delta) drives measure's
    # subtraction to 0.3125, and the exact-delta equality rejects it. The same
    # equality would reject any additive or multiplicative distortion.
    start_tick = 1.25
    true_delta = 0.25
    inflated_end = start_tick + 1.25 * true_delta  # 1.5625, exact
    monkeypatch.setattr(
        time, "perf_counter", _StepClock((start_tick, inflated_end))
    )
    _, report = measure(lambda: None)
    assert report.wall_seconds != true_delta


def test_measure_peak_uses_real_platform_unit_in_process() -> None:
    # In-process pin on the platform the tests actually run on. fn allocates
    # nothing, so the mark is stable across the call: measure's reported peak
    # must equal the raw ru_maxrss normalized by the real platform, bracketed
    # by independent samples taken immediately before and after. The oracle
    # normalizes with a captured pristine reference, never measure's output, so
    # a measure that hardcodes the linux unit (1024x on darwin) lands far above
    # the upper bracket.
    pristine_normalize = normalize_maxrss
    before = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    _, report = measure(lambda: None)
    after = pristine_normalize(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform
    )
    assert before <= report.peak_rss_bytes <= after


def test_measure_peak_in_process_oracle_catches_unit_overscale(
    monkeypatch,
) -> None:
    # Bidirectional proof for the in-process pin: force measure to over-scale by
    # patching the name it resolves, while the oracle keeps a pristine reference.
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
    # Pin measure's peak to an independently sampled ru_maxrss taken in a fresh
    # subprocess. The child re-reads the mark itself (buffer held alive) and
    # normalizes by hand, so the expected value never routes through measure.
    # Run in a subprocess so the oracle does not depend on the parent process's
    # lifetime high-water mark (suite run order): a prior memory-heavy test can
    # no longer flip this assertion.
    result, peak, post_bytes, baseline = _run_probe_peak_child()
    assert result == 499500
    # Full-output check: the reported peak equals the independent post reading
    # exactly, so unit errors, pre-call sampling, child-process usage, and
    # fabricated constants all fail the same assertion.
    assert peak == post_bytes
    # The buffer must have raised the fresh child's mark well past its baseline;
    # a peak that ignores the allocation cannot clear this floor.
    assert peak >= baseline + _PROBE_RISE_FLOOR


def test_measure_peak_oracle_catches_prefn_sampling() -> None:
    # Bidirectional proof, under-report direction: a measure that samples the
    # peak before fn leaves the child's rise at zero, so the independent floor
    # rejects it.
    _, peak, _post, baseline = _run_probe_peak_child(mutate="prefn")
    assert peak < baseline + _PROBE_RISE_FLOOR


def test_measure_peak_oracle_catches_unit_overscale() -> None:
    # Bidirectional proof, over-report direction: a measure that hardcodes the
    # linux kib unit reports 1024x the real darwin bytes, so the exact-equality
    # pin against the independent reading rejects it.
    _, peak, post_bytes, _baseline = _run_probe_peak_child(mutate="overscale")
    assert peak != post_bytes


# ---------------------------------------------------------------------------
# normalize_maxrss branch pins, independent of the run platform.
#
# Every test above reaches only the darwin identity branch, leaving the linux
# scaling, the linux-prefix family, and the unknown-platform guard untested here.
# These call normalize_maxrss with explicit platform strings so the embedded
# suite exercises each branch regardless of the host it runs on. Oracles are the
# stated unit convention applied by hand (darwin bytes pass through; linux kib
# scaled by 1024), never a value read back from normalize_maxrss.
# ---------------------------------------------------------------------------


def test_normalize_maxrss_darwin_identity() -> None:
    assert normalize_maxrss(123456, "darwin") == 123456


def test_normalize_maxrss_linux_scales_by_1024() -> None:
    # Pins the exact linux scale constant: a factor of 512 or 2048 fails here.
    assert normalize_maxrss(123456, "linux") == 123456 * 1024
    assert normalize_maxrss(1, "linux") == 1024


def test_normalize_maxrss_linux_prefix_family_scaled() -> None:
    # The kib->byte contract keys on the "linux" prefix, not the exact literal.
    # A narrowing to `== "linux"` would raise for "linux2" instead of scaling.
    assert normalize_maxrss(7, "linux2") == 7 * 1024


def test_normalize_maxrss_unknown_platform_rejected() -> None:
    # The guard turning a silent wrong-unit reading into a hard error: dropping
    # it (e.g. returning int(ru_maxrss) for any platform) fails here.
    import pytest

    with pytest.raises(ValueError, match="win32"):
        normalize_maxrss(1, "win32")


# ---------------------------------------------------------------------------
# ProbeReport.__post_init__ validation and coercion.
#
# measure() only ever constructs ProbeReport from valid positive readings, so no
# test above crosses a validation boundary. These construct it directly at each
# rejected boundary and pin the wall_seconds float coercion, giving a meaningful signal to the
# "both fields must be strictly positive; a broken probe raises" contract.
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
