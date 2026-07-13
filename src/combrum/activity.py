"""Typed activity events and a human-readable progress formatter."""

from __future__ import annotations

import enum
import sys
from dataclasses import dataclass
from typing import Protocol, TextIO

__all__ = ["ActivityConfig"]


class ActivityLevel(str, enum.Enum):
    OFF = "off"
    SUMMARY = "summary"
    ITERATIONS = "iterations"
    DIAGNOSTIC = "diagnostic"

    @classmethod
    def coerce(cls, value: ActivityLevel | str) -> ActivityLevel:
        if isinstance(value, ActivityLevel):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(level.value for level in cls)
            raise ValueError(
                f"unknown activity level {value!r}; expected one of {choices}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    """Configuration for root-local stdout progress.

    ``level`` is one of ``"off"``, ``"summary"``, ``"iterations"``, or
    ``"diagnostic"``. Set ``stdout=True`` to print progress on the root rank.
    """

    label: str = "combrum"
    run_id: str | None = None
    level: ActivityLevel | str = ActivityLevel.OFF
    stdout: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", ActivityLevel.coerce(self.level))

    @property
    def enabled(self) -> bool:
        return self.level is not ActivityLevel.OFF


def _activity_details(level: ActivityLevel) -> bool:
    return level in {ActivityLevel.ITERATIONS, ActivityLevel.DIAGNOSTIC}


def _object_name(value: object | None) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "__name__", type(value).__name__))


@dataclass(frozen=True, slots=True)
class RowGenStart:
    run_id: str | None = None
    label: str = "rowgen"
    n_obs: int | None = None
    n_simulations: int | None = None
    n_features: int | None = None
    n_parameters: int | None = None
    n_agents: int | None = None
    tolerance: float | None = None
    max_iterations: int | None = None
    min_iterations: int | None = None
    schedule: str | None = None
    cut_policy: str | None = None
    rank: int | None = None
    world_size: int | None = None
    transport: str | None = None
    activity_level: ActivityLevel | str | None = None


@dataclass(frozen=True, slots=True)
class RowGenIteration:
    run_id: str | None = None
    label: str = "rowgen"
    iteration: int = 0
    gap: float | None = None
    gap_delta: float | None = None
    objective: float | None = None
    objective_delta: float | None = None
    active_cuts: int | None = None
    cuts_added: int | None = None
    cuts_dropped: int | None = None
    violation_count: int | None = None
    n_priced_local: int | None = None
    n_inexact_local: int | None = None
    reduce_rounds: int | None = None
    exchange_rounds: int | None = None
    full_sweep: bool | None = None
    convergence_candidate: bool | None = None
    price_seconds: float | None = None
    master_seconds: float | None = None
    comm_seconds: float | None = None
    iteration_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RowGenFinal:
    run_id: str | None = None
    label: str = "rowgen"
    converged: bool | None = None
    termination_reason: str | None = None
    iterations: int | None = None
    final_gap: float | None = None
    objective: float | None = None
    active_cuts: int | None = None
    wall_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BootstrapStart:
    run_id: str | None = None
    label: str = "bootstrap"
    n_bootstrap: int | None = None
    base_seed: int | None = None
    resampling: str | None = None
    tolerance: float | None = None
    max_iterations: int | None = None
    min_iterations: int | None = None
    n_obs: int | None = None
    n_simulations: int | None = None
    n_parameters: int | None = None
    n_agents: int | None = None
    master_backend: str | None = None
    formulation: str | None = None
    cut_policy: str | None = None
    result_publication: str | None = None
    transport: str | None = None
    warm_start: bool | None = None
    rank: int | None = None
    world_size: int | None = None
    activity_level: ActivityLevel | str | None = None
    dual_store_dir: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapRound:
    run_id: str | None = None
    label: str = "bootstrap"
    round_index: int = 0
    live_count: int | None = None
    retired_count: int | None = None
    total_retired: int | None = None
    total_converged: int | None = None
    max_gap: float | None = None
    total_violations: int | None = None
    live_rep_ids: tuple[int, ...] | None = None
    retired_rep_ids: tuple[int, ...] | None = None
    price_seconds: float | None = None
    comm_seconds: float | None = None
    master_seconds: float | None = None
    round_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BootstrapRepFinal:
    run_id: str | None = None
    label: str = "bootstrap"
    rep_id: int = 0
    slot: int | None = None
    owner_rank: int | None = None
    state: str = "computed"
    converged: bool | None = None
    iterations: int | None = None
    final_gap: float | None = None
    objective: float | None = None
    active_cuts: int | None = None
    wall_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BootstrapFinal:
    run_id: str | None = None
    label: str = "bootstrap"
    n_requested: int | None = None
    n_persisted: int | None = None
    n_computed: int | None = None
    n_nonconverged: int | None = None
    n_converged: int | None = None
    total_super_steps: int | None = None
    wall_seconds: float | None = None
    n_duals_stored: int | None = None


ActivityEvent = (
    RowGenStart
    | RowGenIteration
    | RowGenFinal
    | BootstrapStart
    | BootstrapRound
    | BootstrapRepFinal
    | BootstrapFinal
)


class ActivitySink(Protocol):
    def emit(self, event: ActivityEvent) -> None: ...


@dataclass(slots=True)
class ActivityRun:
    config: ActivityConfig
    sink: ActivitySink | None = None

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    def emit(self, event: ActivityEvent) -> None:
        if self.sink is not None:
            self.sink.emit(event)

    def close(self) -> None:
        """No persistent sinks to release."""

    def __enter__(self) -> ActivityRun:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def build_activity_run(
    config: ActivityConfig | None,
    *,
    is_root: bool,
    stream: TextIO | None = None,
) -> ActivityRun:
    cfg = config if config is not None else ActivityConfig()
    if not cfg.enabled or not is_root or not cfg.stdout:
        return ActivityRun(config=cfg)
    sink = SafeActivitySink(RootTableFormatter(stream=stream, label=cfg.label))
    return ActivityRun(config=cfg, sink=sink)


class SafeActivitySink:
    """Disable failing child sinks without escaping into an MPI loop."""

    def __init__(
        self,
        *sinks: ActivitySink,
        warn: TextIO | None = None,
    ) -> None:
        self._sinks = list(sinks)
        self._active = [True] * len(self._sinks)
        self._warn = sys.stderr if warn is None else warn
        self._warned = False
        self._failures: list[str] = []

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(self._failures)

    @property
    def failed(self) -> bool:
        return bool(self._failures)

    def emit(self, event: ActivityEvent) -> None:
        for index, sink in enumerate(self._sinks):
            if not self._active[index]:
                continue
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001
                self._active[index] = False
                message = f"{type(sink).__name__}: {type(exc).__name__}: {exc}"
                self._failures.append(message)
                if not self._warned:
                    self._warned = True
                    print(
                        f"[activity] disabled failing sink: {message}",
                        file=self._warn,
                    )


class RootTableFormatter:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        label: str | None = None,
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.label = label
        self._rowgen_rows = 0
        self._round_rows = 0
        self._rep_rows = 0
        self._rowgen_headers: tuple[str, ...] | None = None
        self._round_headers: tuple[str, ...] | None = None
        self._rep_headers: tuple[str, ...] | None = None
        self._last_gap: float | None = None
        self._last_objective: float | None = None

    def emit(self, event: ActivityEvent) -> None:
        if isinstance(event, RowGenStart):
            self._rowgen_start(event)
        elif isinstance(event, RowGenIteration):
            self._rowgen_iteration(event)
        elif isinstance(event, RowGenFinal):
            self._rowgen_final(event)
        elif isinstance(event, BootstrapStart):
            self._bootstrap_start(event)
        elif isinstance(event, BootstrapRound):
            self._bootstrap_round(event)
        elif isinstance(event, BootstrapRepFinal):
            self._bootstrap_rep(event)
        elif isinstance(event, BootstrapFinal):
            self._bootstrap_final(event)

    def _write(self, label: str, text: str) -> None:
        self.stream.write(f"[{label}] {text}\n")
        self.stream.flush()

    def _event_label(self, event: ActivityEvent) -> str:
        return str(event.label or self.label or "combrum")

    def _rowgen_start(self, event: RowGenStart) -> None:
        parts: list[str] = ["row generation:"]
        parts.extend(
            _named_values(
                ("N", event.n_obs),
                ("S", event.n_simulations),
                ("M", event.n_features),
                ("K", event.n_parameters),
                ("agents", event.n_agents),
                ("ranks", event.world_size),
                ("tol", _fmt_sci(event.tolerance)),
                ("max_iter", event.max_iterations),
                ("cuts", event.cut_policy),
            )
        )
        self._write(self._event_label(event), " ".join(parts))

    def _rowgen_iteration(self, event: RowGenIteration) -> None:
        gap_delta = event.gap_delta
        if gap_delta is None and event.gap is not None and self._last_gap is not None:
            gap_delta = event.gap - self._last_gap
        objective_delta = event.objective_delta
        if (
            objective_delta is None
            and event.objective is not None
            and self._last_objective is not None
        ):
            objective_delta = event.objective - self._last_objective

        cols = [("iter", str(event.iteration))]
        if event.gap is not None:
            cols.append(("gap", _fmt_sci(event.gap)))
            cols.append(("dgap", "-" if gap_delta is None else _fmt_sci(gap_delta)))
        elif gap_delta is not None:
            cols.append(("dgap", _fmt_sci(gap_delta)))
        if event.objective is not None:
            cols.append(("obj", _fmt_sci(event.objective)))
            cols.append(
                ("dobj", "-" if objective_delta is None else _fmt_sci(objective_delta))
            )
        elif objective_delta is not None:
            cols.append(("dobj", _fmt_sci(objective_delta)))
        cols.extend(
            _optional_columns(
                ("+cuts", event.cuts_added),
                ("-cuts", event.cuts_dropped),
                ("cuts", event.active_cuts),
                ("priced", event.n_priced_local),
                ("inexact", event.n_inexact_local),
                ("price", _fmt_time(event.price_seconds)),
                ("master", _fmt_time(event.master_seconds)),
                ("comm", _fmt_time(event.comm_seconds)),
                ("dt", _fmt_time(event.iteration_seconds)),
                ("total", _fmt_time(event.total_seconds)),
            )
        )
        headers = tuple(header for header, _ in cols)
        if self._should_print_header(self._rowgen_rows, self._rowgen_headers, headers):
            self._write(self._event_label(event), _format_header(cols))
            self._rowgen_headers = headers
        self._write(self._event_label(event), _format_values(cols))
        self._rowgen_rows += 1
        if event.gap is not None:
            self._last_gap = event.gap
        if event.objective is not None:
            self._last_objective = event.objective

    def _rowgen_final(self, event: RowGenFinal) -> None:
        parts = ["done"]
        parts.extend(
            _named_values(
                ("converged", _yesno(event.converged)),
                ("reason", event.termination_reason),
                ("iters", event.iterations),
                ("gap", _fmt_sci(event.final_gap)),
                ("cuts", event.active_cuts),
                ("obj", _fmt_sci(event.objective)),
                ("wall", _fmt_time(event.wall_seconds)),
            )
        )
        self._write(self._event_label(event), " ".join(parts))

    def _bootstrap_start(self, event: BootstrapStart) -> None:
        parts = ["bootstrap:"]
        parts.extend(
            _named_values(
                ("reps", event.n_bootstrap),
                ("ranks", event.world_size),
                ("method", event.resampling),
                ("N", event.n_obs),
                ("S", event.n_simulations),
                ("K", event.n_parameters),
                ("agents", event.n_agents),
                ("backend", event.master_backend),
                ("formulation", event.formulation),
                ("policy", event.cut_policy),
                ("publish", event.result_publication),
                ("tol", _fmt_sci(event.tolerance)),
                ("max_iter", event.max_iterations),
                ("min_iter", event.min_iterations),
                ("seed", event.base_seed),
                ("store", event.dual_store_dir),
            )
        )
        self._write(self._event_label(event), " ".join(parts))

    def _bootstrap_round(self, event: BootstrapRound) -> None:
        cols = [("round", str(event.round_index))]
        cols.extend(
            _optional_columns(
                ("live", event.live_count),
                ("retired", event.retired_count),
                ("conv", event.total_converged),
                ("price", _fmt_time(event.price_seconds)),
                ("comm", _fmt_time(event.comm_seconds)),
                ("master", _fmt_time(event.master_seconds)),
                ("dt", _fmt_time(event.round_seconds)),
                ("max_gap", _fmt_sci(event.max_gap)),
                ("viols", event.total_violations),
                ("total", _fmt_time(event.total_seconds)),
            )
        )
        headers = tuple(header for header, _ in cols)
        if self._should_print_header(self._round_rows, self._round_headers, headers):
            self._write(self._event_label(event), _format_header(cols))
            self._round_headers = headers
        self._write(self._event_label(event), _format_values(cols))
        self._round_rows += 1

    def _bootstrap_rep(self, event: BootstrapRepFinal) -> None:
        cols = [("rep", str(event.rep_id))]
        cols.extend(
            _optional_columns(
                ("slot", event.slot),
                ("state", event.state),
                ("conv", _yesno(event.converged)),
                ("iters", event.iterations),
                ("gap", _fmt_sci(event.final_gap)),
                ("cuts", event.active_cuts),
                ("obj", _fmt_sci(event.objective)),
                ("wall", _fmt_time(event.wall_seconds)),
            )
        )
        headers = tuple(header for header, _ in cols)
        if self._should_print_header(self._rep_rows, self._rep_headers, headers):
            self._write(self._event_label(event), _format_header(cols))
            self._rep_headers = headers
        self._write(self._event_label(event), _format_values(cols))
        self._rep_rows += 1

    def _bootstrap_final(self, event: BootstrapFinal) -> None:
        parts = ["done"]
        if event.n_converged is not None and event.n_requested is not None:
            parts.append(f"converged={event.n_converged}/{event.n_requested}")
        parts.extend(
            _named_values(
                ("persisted", event.n_persisted),
                ("computed", event.n_computed),
                ("nonconverged", event.n_nonconverged),
                ("super_steps", event.total_super_steps),
                ("wall", _fmt_time(event.wall_seconds)),
                ("stored", event.n_duals_stored),
            )
        )
        self._write(self._event_label(event), " ".join(parts))

    def _should_print_header(
        self,
        row_count: int,
        previous: tuple[str, ...] | None,
        current: tuple[str, ...],
    ) -> bool:
        return previous != current or row_count % 80 == 0


def _optional_columns(
    *items: tuple[str, object | None],
) -> list[tuple[str, str]]:
    cols: list[tuple[str, str]] = []
    for name, value in items:
        if value is None:
            continue
        cols.append((name, str(value)))
    return cols


def _named_values(*items: tuple[str, object | None]) -> list[str]:
    parts: list[str] = []
    for name, value in items:
        if value is None:
            continue
        parts.append(f"{name}={value}")
    return parts


_MIN_COLUMN_WIDTHS = {
    "iter": 4,
    "round": 5,
    "rep": 3,
    "slot": 4,
    "state": 9,
    "conv": 4,
    "gap": 10,
    "dgap": 10,
    "max_gap": 10,
    "obj": 10,
    "dobj": 10,
    "+cuts": 5,
    "-cuts": 5,
    "cuts": 5,
    "priced": 6,
    "inexact": 7,
    "price": 6,
    "master": 6,
    "comm": 5,
    "dt": 6,
    "wall": 7,
    "total": 7,
    "live": 5,
    "retired": 7,
    "viols": 5,
    "iters": 5,
}


def _format_header(cols: list[tuple[str, str]]) -> str:
    return _format_table_line(cols, use_header=True)


def _format_values(cols: list[tuple[str, str]]) -> str:
    return _format_table_line(cols, use_header=False)


def _format_table_line(cols: list[tuple[str, str]], *, use_header: bool) -> str:
    widths = [
        max(_MIN_COLUMN_WIDTHS.get(header, 0), len(header), len(value))
        for header, value in cols
    ]
    chosen = [header if use_header else value for header, value in cols]
    return "  ".join(value.rjust(width) for value, width in zip(chosen, widths))


def _fmt_sci(value: float | None) -> str | None:
    return None if value is None else f"{float(value):.3e}"


def _fmt_time(value: float | None) -> str | None:
    return None if value is None else f"{float(value):.2f}s"


def _yesno(value: bool | None) -> str | None:
    if value is None:
        return None
    return "yes" if value else "no"
