"""Tests for the activity progress log."""

from __future__ import annotations

import io

import pytest

from combrum.activity import (
    ActivityConfig,
    ActivityLevel,
    BootstrapFinal,
    BootstrapRepFinal,
    RootTableFormatter,
    RowGenFinal,
    RowGenIteration,
    RowGenStart,
    SafeActivitySink,
)


def test_activity_config_defaults_off_and_coerces_level() -> None:
    cfg = ActivityConfig()
    assert cfg.level is ActivityLevel.OFF
    assert not cfg.enabled

    summary = ActivityConfig(level="summary", stdout=True)
    assert summary.enabled
    assert summary.level is ActivityLevel.SUMMARY

    # enabled depends on level alone, not stdout.
    for level in ("off", "summary", "iterations", "diagnostic"):
        expected = level != "off"
        for stdout in (True, False):
            cfg_lvl = ActivityConfig(level=level, stdout=stdout)
            assert cfg_lvl.enabled is expected, (level, stdout)

    with pytest.raises(ValueError, match="unknown activity level"):
        ActivityConfig(level="nope")


def test_root_table_formatter_elides_absent_rowgen_column_groups() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(RowGenIteration(label="toy", iteration=1, gap=0.25))
    formatter.emit(
        RowGenFinal(
            label="toy",
            converged=True,
            iterations=2,
            final_gap=0.0,
            objective=-1.0,
            wall_seconds=3.0,
        )
    )

    lines = stream.getvalue().splitlines()
    assert lines == [
        "[toy] iter         gap        dgap",
        "[toy]    1   2.500e-01           -",
        "[toy] done converged=yes iters=2 gap=0.000e+00 obj=-1.000e+00 wall=3.00s",
    ]
    assert "obj" not in lines[0]
    assert "master" not in lines[0]
    assert all(line.isascii() for line in lines)


def test_root_table_formatter_uses_stable_rowgen_delta_columns() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(
        RowGenIteration(
            label="toy",
            iteration=0,
            gap=0.25,
            objective=-10.0,
            cuts_added=3,
            active_cuts=3,
        )
    )
    formatter.emit(
        RowGenIteration(
            label="toy",
            iteration=1,
            gap=0.10,
            objective=-8.0,
            cuts_added=2,
            active_cuts=5,
        )
    )

    assert stream.getvalue().splitlines() == [
        "[toy] iter         gap        dgap         obj        dobj  +cuts   cuts",
        "[toy]    0   2.500e-01           -  -1.000e+01           -      3      3",
        "[toy]    1   1.000e-01  -1.500e-01  -8.000e+00   2.000e+00      2      5",
    ]


def test_root_table_formatter_obj_min_width_pads_short_positive_objective() -> None:
    # A positive objective formats to 9 chars ("1.000e+00") and the "obj"
    # header to 3, both padded to the obj/dobj min-width of 10. Negative
    # objectives fill all 10 chars and would mask the padding, so the first
    # row carries a positive objective.
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(
        RowGenIteration(
            label="toy", iteration=0, gap=0.25, objective=1.0, cuts_added=3
        )
    )
    formatter.emit(
        RowGenIteration(
            label="toy", iteration=1, gap=0.10, objective=-8.0, cuts_added=2
        )
    )

    assert stream.getvalue().splitlines() == [
        "[toy] iter         gap        dgap         obj        dobj  +cuts",
        "[toy]    0   2.500e-01           -   1.000e+00           -      3",
        "[toy]    1   1.000e-01  -1.500e-01  -8.000e+00  -9.000e+00      2",
    ]


def test_root_table_formatter_reprints_header_when_columns_change() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    # First iteration carries gap only; second adds obj/dobj/+cuts, so the
    # column set changes and the header must be reprinted for the wider table.
    formatter.emit(RowGenIteration(label="toy", iteration=0, gap=0.25))
    formatter.emit(
        RowGenIteration(
            label="toy", iteration=1, gap=0.10, objective=-8.0, cuts_added=2
        )
    )

    # dobj is "-" because the first row carried no objective to diff against.
    assert stream.getvalue().splitlines() == [
        "[toy] iter         gap        dgap",
        "[toy]    0   2.500e-01           -",
        "[toy] iter         gap        dgap         obj        dobj  +cuts",
        "[toy]    1   1.000e-01  -1.500e-01  -8.000e+00           -      2",
    ]


def test_root_table_formatter_reprints_header_every_80_rows() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    # 81 identical-column rows: the header prints before row 0 and again before
    # the 81st data row (row_count 80), independent of any column-set change.
    for i in range(81):
        formatter.emit(RowGenIteration(label="toy", iteration=i, gap=0.5))

    lines = stream.getvalue().splitlines()
    header = "[toy] iter         gap        dgap"
    header_positions = [i for i, line in enumerate(lines) if line == header]
    # One header up top, one at the periodic 80-row boundary; 81 data rows means
    # 83 total lines.
    assert header_positions == [0, 81]
    assert len(lines) == 83


def test_root_table_formatter_bootstrap_header_rep_and_final_rows() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(
        RowGenStart(
            label="auction",
            n_obs=252,
            n_simulations=20,
            n_features=8,
            n_parameters=12,
            n_agents=5040,
            world_size=960,
            tolerance=1e-8,
            max_iterations=200,
            cut_policy="slackstrip",
        )
    )
    formatter.emit(
        BootstrapRepFinal(
            label="boot",
            rep_id=17,
            slot=3,
            state="computed",
            converged=True,
            iterations=31,
            final_gap=7.8e-9,
            objective=-811.0,
            active_cuts=1260,
            wall_seconds=52.3,
        )
    )
    formatter.emit(
        BootstrapFinal(
            label="boot",
            n_requested=400,
            n_converged=397,
            n_persisted=40,
            n_computed=360,
            n_nonconverged=3,
            total_super_steps=73,
            wall_seconds=19440.0,
            n_duals_stored=397,
        )
    )

    lines = stream.getvalue().splitlines()
    assert lines[0] == (
        "[auction] row generation: N=252 S=20 M=8 K=12 agents=5040 "
        "ranks=960 tol=1.000e-08 max_iter=200 cuts=slackstrip"
    )
    assert lines[1] == (
        "[boot] rep  slot      state  conv  iters         gap   cuts"
        "         obj     wall"
    )
    assert lines[2] == (
        "[boot]  17     3   computed   yes     31   7.800e-09   1260"
        "  -8.110e+02   52.30s"
    )
    assert lines[3] == (
        "[boot] done converged=397/400 persisted=40 computed=360 "
        "nonconverged=3 super_steps=73 wall=19440.00s stored=397"
    )
    assert all(line.isascii() for line in lines)


def test_root_table_formatter_renders_nonconverged_rowgen_final() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(RowGenIteration(label="toy", iteration=1, gap=0.25))
    formatter.emit(
        RowGenFinal(
            label="toy",
            converged=False,
            iterations=2,
            final_gap=0.0,
            objective=-1.0,
            wall_seconds=3.0,
        )
    )

    assert stream.getvalue().splitlines()[-1] == (
        "[toy] done converged=no iters=2 gap=0.000e+00 obj=-1.000e+00 wall=3.00s"
    )


def test_root_table_formatter_renders_nonconverged_bootstrap_rep() -> None:
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(
        BootstrapRepFinal(
            label="boot",
            rep_id=17,
            slot=3,
            state="computed",
            converged=False,
            iterations=31,
            final_gap=7.8e-9,
            objective=-811.0,
            active_cuts=1260,
            wall_seconds=52.3,
        )
    )

    lines = stream.getvalue().splitlines()
    assert lines[0] == (
        "[boot] rep  slot      state  conv  iters         gap   cuts"
        "         obj     wall"
    )
    assert lines[1] == (
        "[boot]  17     3   computed    no     31   7.800e-09   1260"
        "  -8.110e+02   52.30s"
    )


@pytest.mark.parametrize(
    ("n_requested", "n_converged", "stray_fragment"),
    [
        # n_requested missing -> would print "397/None" without the guard.
        (None, 397, "converged=397/None"),
        # n_converged missing -> would print "None/400" without the guard.
        (400, None, "converged=None/400"),
    ],
)
def test_root_table_formatter_bootstrap_converged_pair_is_all_or_nothing(
    n_requested: int | None, n_converged: int | None, stray_fragment: str
) -> None:
    # With either count missing the "converged=X/Y" fragment is omitted
    # entirely, never printed with a "None" side.
    stream = io.StringIO()
    formatter = RootTableFormatter(stream=stream)

    formatter.emit(
        BootstrapFinal(
            label="boot",
            n_requested=n_requested,
            n_converged=n_converged,
            n_persisted=40,
            n_computed=360,
            n_nonconverged=3,
            total_super_steps=73,
            wall_seconds=19440.0,
            n_duals_stored=397,
        )
    )

    line = stream.getvalue().splitlines()[0]
    # Leading space avoids a false match on the "nonconverged=" token.
    assert " converged=" not in line
    assert stray_fragment not in line
    # With the fragment dropped the line is identical for both parametrizations.
    assert line == (
        "[boot] done persisted=40 computed=360 nonconverged=3 "
        "super_steps=73 wall=19440.00s stored=397"
    )


def test_safe_activity_sink_disables_only_failing_children() -> None:
    class BadSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, event) -> None:
            self.calls += 1
            raise RuntimeError("boom")

    class SecondBadSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, event) -> None:
            self.calls += 1
            raise ValueError("kaboom")

    class GoodSink:
        def __init__(self) -> None:
            self.events = []

        def emit(self, event) -> None:
            self.events.append(event)

    bad = BadSink()
    bad2 = SecondBadSink()
    good = GoodSink()
    warn = io.StringIO()
    safe = SafeActivitySink(bad, bad2, good, warn=warn)

    safe.emit(RowGenStart(label="toy"))
    safe.emit(RowGenFinal(label="toy", converged=True))

    # Each failing sink is disabled after its first raise, so it is called once
    # and never again; the good sink still receives both events.
    assert bad.calls == 1
    assert bad2.calls == 1
    assert len(good.events) == 2
    assert safe.failed

    assert safe.failures == (
        "BadSink: RuntimeError: boom",
        "SecondBadSink: ValueError: kaboom",
    )

    # The warn-once guard prints only the first failure's warning; later
    # failures are silent.
    assert warn.getvalue().count("disabled failing sink") == 1
    assert "BadSink: RuntimeError: boom" in warn.getvalue()
    assert "SecondBadSink: ValueError: kaboom" not in warn.getvalue()
