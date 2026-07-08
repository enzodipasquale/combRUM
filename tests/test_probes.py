"""RSS-unit tests for ``combrum.runinfo``.

Covers ``normalize_maxrss`` and ``peak_rss_bytes`` -- platform branches and
the kibibyte convention (1 KiB = 1024 bytes).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import combrum.runinfo as runinfo
from combrum.runinfo import normalize_maxrss, peak_rss_bytes


# --- normalize_maxrss unit convention (sys.platform-driven) -----------------


@pytest.mark.parametrize(
    "fake_platform, raw, expected",
    [
        # darwin ru_maxrss is already bytes -> identity passthrough.
        ("darwin", 123456, 123456),
        ("darwin", 0, 0),
        ("darwin", 1, 1),
        # linux ru_maxrss is kibibytes -> x1024.
        ("linux", 123456, 123456 * 1024),
        ("linux", 1, 1024),
        # the linux branch matches by prefix (startswith), so "linux2" scales too.
        ("linux2", 7, 7 * 1024),
    ],
)
def test_normalize_maxrss_unit_convention(
    monkeypatch: pytest.MonkeyPatch, fake_platform: str, raw: int, expected: int
) -> None:
    monkeypatch.setattr(runinfo.sys, "platform", fake_platform)
    assert normalize_maxrss(raw) == expected


@pytest.mark.parametrize("fake_platform", ["win32", "cygwin", "freebsd12"])
def test_normalize_maxrss_rejects_unknown_platform(
    monkeypatch: pytest.MonkeyPatch, fake_platform: str
) -> None:
    # unknown platforms must raise, not silently return a wrong-unit reading.
    monkeypatch.setattr(runinfo.sys, "platform", fake_platform)
    with pytest.raises(RuntimeError, match="unit convention unknown"):
        normalize_maxrss(1000)
    with pytest.raises(RuntimeError, match=fake_platform):
        normalize_maxrss(1000)


# --- peak_rss_bytes: platform dispatch + RUSAGE_SELF ------------------------


class _FakeRusage:
    """A getrusage return with a fixed ru_maxrss and nothing else."""

    def __init__(self, ru_maxrss: int) -> None:
        self.ru_maxrss = ru_maxrss


def _patched_peak(
    monkeypatch: pytest.MonkeyPatch, platform: str, raw: int
) -> tuple[int, int]:
    """Drive peak_rss_bytes with a known raw reading on a chosen platform.

    Returns (reported_bytes, who) where who is the rusage target
    peak_rss_bytes asked for.
    """
    seen: dict[str, int] = {}

    def fake_getrusage(who: int) -> _FakeRusage:
        seen["who"] = who
        return _FakeRusage(raw)

    monkeypatch.setattr(runinfo.sys, "platform", platform)
    monkeypatch.setattr(runinfo.resource, "getrusage", fake_getrusage)
    reported = peak_rss_bytes()
    return reported, seen["who"]


def test_peak_rss_bytes_linux_scales_and_reads_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # force the linux branch: on a darwin dev/CI host normalize_maxrss is
    # identity, so the kib->byte scaling is never exercised natively.
    raw_kib = 4096
    reported, who = _patched_peak(monkeypatch, "linux", raw_kib)
    assert reported == raw_kib * 1024
    assert who == resource.RUSAGE_SELF


def test_peak_rss_bytes_darwin_identity_and_reads_self(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # darwin branch is identity; paired with the linux case above, dispatch
    # has to read the live sys.platform.
    raw = 7777
    reported, who = _patched_peak(monkeypatch, "darwin", raw)
    assert reported == raw
    assert who == resource.RUSAGE_SELF


# --- peak_rss_bytes: reflects a real allocation, unit-correct ---------------

# 256 MiB dwarfs a fresh interpreter's resident set, so touching every page
# raises the child's lifetime ru_maxrss mark well past its own baseline.
_PROBE_ALLOC_BYTES = 256 * 1024 * 1024
# floor with headroom below the allocation; a kibibyte-unit reading lands
# ~1024x too small and misses it.
_PROBE_RISE_FLOOR = 64 * 1024 * 1024

_PROBE_SRC = textwrap.dedent(
    """
    import resource
    from combrum.runinfo import peak_rss_bytes, normalize_maxrss

    before = normalize_maxrss(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    n_bytes = {alloc}
    buf = bytearray(n_bytes)
    for i in range(0, n_bytes, 4096):
        buf[i] = 1
    reported = peak_rss_bytes()
    floor = before + {floor}
    assert reported >= floor, (before, reported, floor)
    # Bytes, not kibibytes: the peak must be at least the buffer we allocated.
    assert reported >= n_bytes, (reported, n_bytes)
    print("OK")
    """
)


def test_peak_rss_bytes_rises_after_large_allocation(tmp_path: Path) -> None:
    """peak_rss_bytes reflects a large allocation, in bytes.

    ru_maxrss is a per-process high-water mark, so this needs a fresh
    subprocess -- within the shared pytest process an earlier test may have
    pushed the mark far above current usage.
    """
    src = Path(__file__).resolve().parents[1] / "src"
    probe = _PROBE_SRC.format(alloc=_PROBE_ALLOC_BYTES, floor=_PROBE_RISE_FLOOR)
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        cwd=tmp_path,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("OK")
