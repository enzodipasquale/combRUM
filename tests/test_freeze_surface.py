"""Frozen runtime imports for the public package modules.

The table below records each module's intra-package runtime imports. A new
edge is an architecture change and should be added here explicitly. Typing-only
imports (under ``if TYPE_CHECKING:``) are excluded because they add no runtime
coupling.
"""

from __future__ import annotations

import ast
from pathlib import Path

import combrum

# Locate the package from the imported module, not a fixed repo layout, so the
# gate holds whether these tests sit beside src/ or run against an install.
SRC = Path(combrum.__file__).resolve().parent


def _module_files() -> frozenset[str]:
    # Every dotted module name that exists as a real ``.py`` under the package.
    # ``from combrum.pkg import name`` couples to ``combrum.pkg.name`` only when
    # that dotted path is a real module file (a submodule import), not when
    # ``name`` is a re-exported function or class.
    names: set[str] = set()
    for py in SRC.rglob("*.py"):
        rel = py.relative_to(SRC.parent)
        names.add(
            ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
        )
    return frozenset(names)


MODULE_FILES = _module_files()

# The frozen graph. Keys are module names; values are the complete set of
# intra-package modules each one may import at runtime.
FROZEN_IMPORTS: dict[str, frozenset[str]] = {
    "combrum": frozenset(
        {
            "combrum.activity",
            "combrum._version",
            "combrum.bootstrap",
            "combrum.bootstrap_distributed",
            "combrum.callbacks",
            "combrum.cut_policies",
            "combrum.demand",
            "combrum.engine",
            # ``from combrum.engine import estimate`` binds the re-exported
            # estimate function, but executing it first imports the
            # combrum.engine.estimate submodule — a real runtime coupling.
            "combrum.engine.estimate",
            "combrum.formulations",
            "combrum.informed_schedule",
            "combrum.model",
            "combrum.oracle",
            "combrum.parameters",
            "combrum.randomness",
            "combrum.result",
            "combrum.runinfo",
            "combrum.schedule",
            "combrum.solver_settings",
            "combrum.transport",
        }
    ),
    "combrum._bundle_key": frozenset(),
    "combrum._version": frozenset(),
    # The standalone activity-log value layer: typed events, bounded recorders,
    # JSONL sidecars, and root-table formatting. It is intentionally a leaf
    # module: no engine, result, or transport imports. The package root
    # re-exports only the value-level config enum for scripts/notebooks.
    "combrum.activity": frozenset(),
    # The serial bootstrap: B cold weighted refits, one replication at a time.
    # It composes the fit surface — build_fit_context + run_fit (engine,
    # engine.driver) threading each rep's per-observation weights — and folds
    # the per-rep theta/converged/dual into the frozen BootstrapResult (result),
    # over the parameter layout result carries (parameters) and the oracle it
    # refits (oracle). The native weight stream is drawn from the
    # placement-invariant per-rep RNG (randomness); the opt-in per-rep dual is
    # re-stamped onto its slot (dual) and streamed one-in-flight to the per-rep
    # store (dualstore) — the writer side of the streaming dual store, consuming
    # the frozen serializer, not widening it. It imports the transport contract
    # (transport, transport.base) and never reaches engine internals or a
    # solver: the fit runs through run_fit and the master lives behind the
    # builder, matching the distributed bootstrap.
    "combrum.bootstrap": frozenset(
        {
            "combrum.dual",
            "combrum.dualstore",
            "combrum.activity",
            "combrum.engine",
            "combrum.engine.agreement",
            "combrum.engine.driver",
            "combrum.engine.observed",
            "combrum.model",
            "combrum.randomness",
            "combrum.result",
            "combrum.transport",
            "combrum.transport.base",
        }
    ),
    # The distributed-bootstrap scheduler drives bounded replica waves, streams
    # opt-in dual payloads, resolves pricing, draws placement-invariant weights,
    # and owns the batched reduce/exchange over the transport ABC. It reaches no
    # solver, and distributed_context owns observed-feature preparation. The
    # dual/dualstore edges are the
    # writer side of the streaming dual store: the frozen serializer is
    # consumed, not widened. The result/parameters edges are the
    # published-result type and the theta layout it carries: the distributed
    # bootstrap publishes the one frozen BootstrapResult (in result),
    # constructing it over a parameter layout. engine.certify is the aggregate
    # pricing-gap certificate, reduced once after all waves have priced.
    "combrum.bootstrap_distributed": frozenset(
        {
            "combrum.context",
            "combrum.activity",
            "combrum.demand",
            "combrum.dual",
            "combrum.dualstore",
            "combrum.engine.agreement",
            "combrum.engine.certify",
            "combrum.interface_resolution",
            "combrum.engine.distributed_context",
            "combrum.formulations",
            "combrum.masters",
            "combrum.model",
            "combrum.oracle",
            "combrum.parameters",
            "combrum.policies",
            "combrum.randomness",
            "combrum.result",
            "combrum.rowgen",
            "combrum.transport.base",
        }
    ),
    "combrum.certification": frozenset(),
    # Rank-agreement helper layer for public distributed controls, guarded
    # rank-local hooks, and warm-start theta tokens. It depends only on the
    # transport ABC inside the package graph it guards.
    "combrum.engine.agreement": frozenset(
        {"combrum._version", "combrum.transport.base"}
    ),
    "combrum.context": frozenset(
        {
            "combrum.master",
            "combrum.policies",
            "combrum.schedule",
            "combrum.transport.base",
        }
    ),
    # The adaptive-timeout callback helpers: a core module
    # that imports the schedule/settings vocabulary it drives — its own
    # typed schedule lives in-module, so the edges are SolverSettings /
    # SolverConfigurable from the neutral solver_settings module
    # and the Oracle type the produced hook is typed against. It applies
    # settings through the capability protocol alone, so it reaches no
    # concrete solver or oracle, and the frozen Oracle ABC is consumed,
    # never widened.
    "combrum.callbacks": frozenset(
        {"combrum.oracle", "combrum.solver_settings"}
    ),
    "combrum.cut_policies": frozenset(
        {"combrum.policies", "combrum.transport.base"}
    ),
    "combrum.demand": frozenset(),
    "combrum.dual": frozenset(),
    # The either-one resolution guard: a core
    # module that resolves a symmetric per-agent | batched surface once
    # and dispatches it. It names only the transport contract (for the
    # rank-agreement round) plus stdlib/numpy; no solver or engine edge.
    # formulation — so the formulations and the oracle can consume the
    # guard without growing a coupling to either.
    "combrum.interface_resolution": frozenset({"combrum.transport.base"}),
    # The production driver subpackage — the engine-owned phase path, single-replication core:
    # the engine that owns the cross-rank reduce/exchange so the formulation
    # stays transport-passive. Its package init re-exports the public
    # estimate API plus the driver / fit-step / certification / persistent-master
    # surface, so it reaches only its own submodules.
    "combrum.engine": frozenset(
        {
            "combrum.engine.context_builder",
            "combrum.engine.driver",
            "combrum.engine.estimate",
            "combrum.engine.persistent",
        }
    ),
    # The shared fit-context builder: the one owner of the estimation context
    # assembly both the point estimate and the bootstrap/sweep drive. It builds
    # the FitContext from user inputs (context), delegates observed-bundle
    # objective construction to engine.observed, reads the formulation set it
    # dispatches u_coef on (formulations), builds the master (masters), takes the
    # parameter layout (parameters), reads the warm-start result anchor (result),
    # names the cut-policy contract (policies), and types the transport +
    # warm-start CutRow contract
    # (transport.base). It composes these contracts; it never reaches a solver
    # or the master directly (only through make_master and the
    # reinstall/extract_cuts primitive).
    # Gate note: engine.observed intentionally owns the either-one edge needed
    # to infer observed phi from priced features only when Model.observed_features
    # is omitted; explicit observed_features remains the phi-only escape hatch.
    "combrum.engine.context_builder": frozenset(
        {
            "combrum.context",
            "combrum.engine.observed",
            "combrum.formulations",
            "combrum.masters",
            "combrum.parameters",
            "combrum.policies",
            "combrum.result",
            "combrum.transport.base",
        }
    ),
    # Distributed context assembly for the split observation/pricing axes:
    # consumes the same context/result envelope as the dense builder, but gets
    # observed feature rows from the distributed observed-feature surface and
    # builds only an owner-rank lazy NSlack master. It names model/parameters
    # through Model, formulation support through formulations, and the transport
    # ABC for observation-axis keyed reductions and owner-rank guarded setup.
    "combrum.engine.distributed_context": frozenset(
        {
            "combrum.context",
            "combrum.engine.agreement",
            "combrum.engine.context_builder",
            "combrum.formulations",
            "combrum.masters",
            "combrum.model",
            "combrum.transport._common",
            "combrum.transport.base",
        }
    ),
    # Observed-bundle objective materialization: consumes the comm-free
    # features either-one resolver to infer phi rows from the active feature map
    # when no explicit observed_features surface is supplied, and reduces the
    # resulting objective/moment over the transport contract. It imports no
    # formulation, master, solver, or bootstrap module.
    "combrum.engine.observed": frozenset(
        {"combrum.interface_resolution", "combrum.transport.base"}
    ),
    # Per-call gap aggregation into the frozen Certification report: a core
    # module that reads the gap off each Demand the price phase produced and
    # reduces the counts/worst-gap across ranks. It names the frozen report
    # type it fills (certification), the demand envelope it reads (demand),
    # and the transport ABC it reduces over (transport.base). It has no solver,
    # master, or formulation edge.
    "combrum.engine.certify": frozenset(
        {
            "combrum.certification",
            "combrum.demand",
            "combrum.transport.base",
        }
    ),
    # The public estimate APIs: serial estimate consumes the shared full-array
    # context builder, while estimate_distributed consumes the split-axis
    # distributed builder and ResultPublication.SUMMARY. The formulations edge
    # is the early public guard that admits NSlack only. Both drive the loop
    # (engine.driver), certify pricing gaps (engine.certify), publish the frozen
    # result (result), and import the transport contract (transport,
    # transport.base).
    "combrum.engine.estimate": frozenset(
        {
            "combrum.activity",
            "combrum.context",
            "combrum.engine.agreement",
            "combrum.engine.certify",
            "combrum.engine.context_builder",
            "combrum.engine.distributed_context",
            "combrum.engine.driver",
            "combrum.formulations",
            "combrum.model",
            "combrum.oracle",
            "combrum.policies",
            "combrum.result",
            "combrum.runinfo",
            "combrum.schedule",
            "combrum.transport",
            "combrum.transport.base",
        }
    ),
    # The driver loop composes the frozen contracts it drives: the geometry it
    # is handed (context), the price method pair it resolves once and routes
    # pricing through (interface_resolution), the engine-owned fit-step
    # (engine.fitstep), the solve contract it publishes
    # (formulation), the driver-owned dual-concentration schedule branch
    # (informed_schedule) and the schedule ABC it reads (schedule), the
    # optional typed activity event sink (activity), the oracle ABC whose price
    # surface it resolves (oracle), and the phase-step protocol it drives
    # (rowgen). No solver or master edge; it touches the master only
    # through the formulation's apply_step.
    "combrum.engine.driver": frozenset(
        {
            "combrum.activity",
            "combrum.context",
            "combrum.engine.agreement",
            "combrum.interface_resolution",
            "combrum.engine.fitstep",
            "combrum.formulation",
            "combrum.informed_schedule",
            "combrum.oracle",
            "combrum.rowgen",
            "combrum.schedule",
        }
    ),
    # The engine-owned one-iteration fit-step: the price/reduce+exchange/
    # finalise/solve phases, transport-passive over the formulation. It
    # names the demand envelope it prices into (demand), the either-one
    # PRICE contract it routes through (interface_resolution), the phase contract +
    # contribution/reduced union it dispatches on by type (rowgen), and the
    # transport ABC it owns the reduce/exchange on (transport.base). No
    # solver, no master, no formulation module; it composes the
    # RowGenStep protocol, never a concrete method.
    "combrum.engine.fitstep": frozenset(
        {
            "combrum.demand",
            "combrum.interface_resolution",
            "combrum.rowgen",
            "combrum.transport.base",
        }
    ),
    # The persistent-master driver: holds one master
    # across an outer ψ search, RHS-rewriting the carried cuts and warm-solving
    # per ψ. It composes the shared builder's reuse hook + observed-objective
    # helper (engine.context_builder), the loop entry with its suppress_close
    # flag (engine.driver), and types its RHS map over the cut row
    # (transport.base). It can lazily default to NSlack, but the NSlack-only
    # guard still uses exact class identity rather than isinstance.
    "combrum.engine.persistent": frozenset(
        {
            "combrum.engine.context_builder",
            "combrum.engine.driver",
            "combrum.formulations.nslack",
            "combrum.transport.base",
        }
    ),
    "combrum.dualstore": frozenset({"combrum.dual"}),
    "combrum.formulation": frozenset(
        {"combrum.context", "combrum.demand"}
    ),
    # The row-generation methods consume frozen contracts only: the solve
    # contract they implement, the geometry/demand/dual payload types,
    # the backend-neutral master ABC, the cut-policy profile vocabulary, the
    # sparse-routing adapter, and the cut envelope. Solver imports must stay out
    # of this module.
    "combrum.formulations": frozenset(
        {
            "combrum.formulations.nslack",
            "combrum.formulations.oneslack",
        }
    ),
    "combrum.formulations.nslack": frozenset(
        {
            "combrum._bundle_key",
            "combrum.context",
            "combrum.demand",
            "combrum.dual",
            "combrum.interface_resolution",
            "combrum.formulation",
            "combrum.master",
            "combrum.policies",
            "combrum.rowgen",
            "combrum.steprecord",
            "combrum.transport.base",
        }
    ),
    "combrum.formulations.oneslack": frozenset(
        {
            "combrum.context",
            "combrum.demand",
            "combrum.interface_resolution",
            "combrum.formulation",
            "combrum.master",
            "combrum.rowgen",
            "combrum.steprecord",
            "combrum.transport.base",
        }
    ),
    "combrum.informed_schedule": frozenset({"combrum.schedule"}),
    "combrum.master": frozenset({"combrum.transport.base"}),
    # The two domain nouns: a plain dataclass module that imports the oracle,
    # parameter, feature-map, and formulation contracts it holds as typed fields.
    # These edges are runtime so Model annotations stay introspectable. No engine
    # edge: this is a value type, not a fit driver.
    "combrum.model": frozenset(
        {
            "combrum.formulation",
            "combrum.formulations",
            "combrum.interface_resolution",
            "combrum.oracle",
            "combrum.parameters",
        }
    ),
    # The factory's backend edges are function-body imports — the loader
    # contract that keeps the package import solver-free — and count as
    # runtime imports here exactly because they execute at call time.
    "combrum.masters": frozenset(
        {
            "combrum.master",
            "combrum.masters.gurobi",
            "combrum.masters.highs",
            "combrum.transport.base",
        }
    ),
    "combrum.masters._common": frozenset(),
    "combrum.masters.gurobi": frozenset(
        {"combrum.master", "combrum.masters._common", "combrum.transport.base"}
    ),
    "combrum.masters.highs": frozenset(
        {"combrum.master", "combrum.masters._common", "combrum.transport.base"}
    ),
    "combrum.oracle": frozenset(
        {"combrum.demand", "combrum.transport.base"}
    ),
    "combrum.parameters": frozenset(),
    "combrum.policies": frozenset({"combrum.transport.base"}),
    "combrum.randomness": frozenset(),
    "combrum.reductions": frozenset(),
    "combrum.result": frozenset(
        {
            "combrum.certification",
            "combrum.dual",
            "combrum.parameters",
            "combrum.runinfo",
            "combrum.transport.base",
        }
    ),
    # The run-metadata surface carries the already-computed diagnostics,
    # certification report, provenance, and node layout. These runtime imports
    # keep public result annotations introspectable; the module still only
    # surfaces data the run already produced, never re-derives it.
    "combrum.runinfo": frozenset(
        {
            "combrum.certification",
            "combrum.engine.driver",
            "combrum.transport.base",
        }
    ),
    # The composable row-generation phase contract: a core
    # module the formulations implement so a future engine owns the
    # cross-rank reduce/exchange. It names only the demand envelope and
    # the cut row that envelope carries; the install phase touches the master
    # through the formulation, not from here.
    "combrum.rowgen": frozenset(
        {"combrum.demand", "combrum.transport.base"}
    ),
    "combrum.schedule": frozenset(),
    # The wholesale-capture record: a core module
    # the formulations emit into so the either-one gate can witness every
    # filter-chain input over its full pre-filter domain. Like rowgen it
    # names only the demand envelope; cut-row identities are stored as typed
    # scalar fields, so it needs no transport import. It captures values the
    # formulation already holds; it never reaches the master itself. The two
    # formulations grow an edge to it (the emission), mirroring the rowgen
    # precedent where a new core phase module is consumed by both.
    "combrum.steprecord": frozenset({"combrum._bundle_key", "combrum.demand"}),
    # The neutral runtime-settings contract: SolverSettings /
    # SolverConfigurable, framework types with no concrete-solver edge, so
    # the callback layer drives them without reaching the solver package.
    "combrum.solver_settings": frozenset(),
    "combrum.transport": frozenset(
        {
            "combrum.transport.base",
            "combrum.transport.mpi",
            "combrum.transport.reference",
        }
    ),
    "combrum.transport._common": frozenset(),
    "combrum.transport.base": frozenset({"combrum._bundle_key"}),
    # The MPI implementation consumes exactly the contracts the references
    # consume: the frozen contract and the shared reduction kernel.
    # mpi4py itself is loaded lazily at instantiation (an optional
    # extra), so it contributes no import edge anywhere.
    "combrum.transport.mpi": frozenset(
        {
            "combrum.reductions",
            "combrum.transport._common",
            "combrum.transport.base",
        }
    ),
    "combrum.transport.reference": frozenset(
        {
            "combrum.reductions",
            "combrum.transport._common",
            "combrum.transport.base",
        }
    ),
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent)
    name = ".".join(rel.with_suffix("").parts)
    return name.removesuffix(".__init__")


def _package_of(path: Path) -> str:
    # The importing module's ``__package__`` — the base a relative import
    # resolves against. A package ``__init__.py`` resolves relative to itself;
    # a plain module resolves relative to its containing package.
    name = _module_name(path)
    if path.name == "__init__.py":
        return name
    return name.rpartition(".")[0]


def _resolve_relative(base_pkg: str, level: int, module: str | None) -> str:
    # Emulate Python's relative-import resolution: level 1 stays in the current
    # package, each extra level strips one trailing component, then the imported
    # module tail (if any) is appended.
    parts = base_pkg.split(".")
    if level - 1 > 0:
        parts = parts[: -(level - 1)]
    root = ".".join(parts)
    if module:
        return f"{root}.{module}" if root else module
    return root


def _is_type_checking_test(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _from_edges(
    node: ast.ImportFrom, base_pkg: str, module_files: frozenset[str]
) -> set[str]:
    # The intra-package edges a single ``from ... import ...`` statement adds.
    if not node.level:
        # Absolute intra-package import: ``from combrum.x import ...``.
        if not (node.module and node.module.startswith("combrum")):
            return set()
        edges = {node.module}
        # ``from combrum.pkg import sub`` couples to the submodule combrum.pkg.sub
        # when that is a real module file, mirroring the relative branch's
        # ``from . import a, b`` handling. A re-exported function/class name has
        # no matching module file and adds no edge.
        for alias in node.names:
            candidate = f"{node.module}.{alias.name}"
            if candidate in module_files:
                edges.add(candidate)
        return edges
    # Relative import: resolve against this module's own package so a new
    # coupling introduced as ``from . import sibling`` or
    # ``from .sibling import name`` is seen exactly like the absolute form.
    resolved = _resolve_relative(base_pkg, node.level, node.module)
    if not resolved.startswith("combrum"):
        return set()
    if node.module:
        # ``from .sibling import name`` — the coupled module is the resolved
        # package tail, matching the absolute branch which records the
        # imported-from module, not the leaf names.
        return {resolved}
    # ``from . import a, b`` — each name is itself a submodule edge.
    return {f"{resolved}.{alias.name}" for alias in node.names}


def _runtime_imports(
    path: Path, *, base_pkg: str | None = None
) -> frozenset[str]:
    found: set[str] = set()
    base_pkg = _package_of(path) if base_pkg is None else base_pkg

    def visit(nodes: list[ast.stmt], type_checking: bool) -> None:
        for node in nodes:
            if isinstance(node, ast.If):
                guarded = type_checking or _is_type_checking_test(node.test)
                visit(node.body, guarded)
                visit(node.orelse, type_checking)
                continue
            if isinstance(node, ast.Match):
                # ast.Match keeps its arms in .cases (not a body/orelse field),
                # so recurse explicitly or an import relocated into a case arm
                # would vanish from the graph unseen.
                for case in node.cases:
                    visit(case.body, type_checking)
                continue
            if not type_checking:
                if isinstance(node, ast.ImportFrom):
                    found.update(_from_edges(node, base_pkg, MODULE_FILES))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("combrum"):
                            found.add(alias.name)
            for field in ("body", "orelse", "finalbody", "handlers"):
                children = getattr(node, field, None)
                if isinstance(children, list) and not isinstance(node, ast.If):
                    stmts = [
                        c
                        for c in children
                        if isinstance(c, (ast.stmt, ast.excepthandler))
                    ]
                    for child in stmts:
                        if isinstance(child, ast.excepthandler):
                            visit(child.body, type_checking)
                    visit(
                        [c for c in stmts if isinstance(c, ast.stmt)],
                        type_checking,
                    )

    visit(ast.parse(path.read_text()).body, type_checking=False)
    return frozenset(found)


def test_runtime_import_graph_is_frozen() -> None:
    actual = {
        _module_name(py): _runtime_imports(py) for py in SRC.rglob("*.py")
    }
    assert set(actual) == set(FROZEN_IMPORTS), (
        "module set changed: "
        f"new={sorted(set(actual) - set(FROZEN_IMPORTS))}, "
        f"gone={sorted(set(FROZEN_IMPORTS) - set(actual))} — a contract "
        "module appeared or vanished; amend FROZEN_IMPORTS explicitly"
    )
    for module, expected in FROZEN_IMPORTS.items():
        got = actual[module]
        assert got == expected, (
            f"{module}: runtime import surface changed — "
            f"added={sorted(got - expected)}, removed={sorted(expected - got)}"
            " — a new intra-package coupling is an architecture change, not"
            " an implementation detail"
        )


def _parse_from(source: str) -> ast.ImportFrom:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.ImportFrom)
    return node


def test_from_package_import_submodule_is_recorded() -> None:
    # ``from combrum.engine import driver`` couples to the real submodule
    # combrum.engine.driver, not just the parent package — otherwise a new
    # submodule import hides inside an already-frozen package edge.
    node = _parse_from("from combrum.engine import driver")
    edges = _from_edges(node, base_pkg="combrum", module_files=MODULE_FILES)
    assert "combrum.engine.driver" in edges
    assert "combrum.engine" in edges


def test_from_package_import_reexported_name_adds_no_submodule_edge() -> None:
    # ``build_fit_context`` is a re-exported function, not a module file, so it
    # must not fabricate a spurious combrum.engine.build_fit_context edge.
    node = _parse_from("from combrum.engine import build_fit_context")
    edges = _from_edges(node, base_pkg="combrum", module_files=MODULE_FILES)
    assert edges == {"combrum.engine"}


def test_relative_from_edges_resolve_against_own_package() -> None:
    # Current src uses only absolute imports, so the ``node.level > 0`` branch of
    # _from_edges (and _resolve_relative behind it) is otherwise never exercised.
    # Drive it directly with hand-resolved expecteds derived from Python's
    # relative-import rules — strip ``level - 1`` trailing components of the
    # importing package, then attach the module tail (or each bare name) — so a
    # future src module that adopts ``from . import sibling`` is attributed to the
    # right absolute edge. Expecteds are NOT read back from _resolve_relative.
    #
    # ``from . import demand`` inside package ``combrum``: level 1 stays in the
    # package, each bare name is its own submodule edge.
    node = _parse_from("from . import demand")
    assert _from_edges(node, base_pkg="combrum", module_files=MODULE_FILES) == {
        "combrum.demand"
    }
    # Multiple bare names each become an edge.
    node = _parse_from("from . import demand, oracle")
    assert _from_edges(node, base_pkg="combrum", module_files=MODULE_FILES) == {
        "combrum.demand",
        "combrum.oracle",
    }
    # ``from .transport import base`` records the from-module (combrum.transport),
    # mirroring the absolute branch which records the imported-from module.
    node = _parse_from("from .transport import base")
    assert _from_edges(node, base_pkg="combrum", module_files=MODULE_FILES) == {
        "combrum.transport"
    }
    # Multi-level: ``from ..`` inside combrum.engine strips one component back to
    # combrum, so a bare name resolves to combrum.x — this pins the level-1
    # stripping arithmetic that a level-0 file could never reach.
    node = _parse_from("from .. import x")
    assert _from_edges(
        node, base_pkg="combrum.engine", module_files=MODULE_FILES
    ) == {"combrum.x"}
    # ``from ..pkg import y`` inside combrum.engine -> from-module combrum.pkg.
    node = _parse_from("from ..pkg import y")
    assert _from_edges(
        node, base_pkg="combrum.engine", module_files=MODULE_FILES
    ) == {"combrum.pkg"}
    # Three levels up from combrum.engine.driver strips two components, then the
    # dotted tail ``a.b`` attaches -> combrum.a.b.
    node = _parse_from("from ...a.b import c")
    assert _from_edges(
        node, base_pkg="combrum.engine.driver", module_files=MODULE_FILES
    ) == {"combrum.a.b"}


def test_relative_from_edges_outside_package_add_no_edge() -> None:
    # A relative import that resolves outside combrum contributes no edge, the
    # relative-branch counterpart of the absolute non-combrum guard.
    node = _parse_from("from . import something")
    assert (
        _from_edges(node, base_pkg="notcombrum", module_files=MODULE_FILES)
        == set()
    )


def test_walker_sees_imports_inside_match_case_arms(tmp_path: Path) -> None:
    # ast.Match arms live in .cases, outside the body/orelse fields the walker
    # descends by default; an import relocated into a case must still register.
    source = (
        "def f(x):\n"
        "    match x:\n"
        "        case 1:\n"
        "            from combrum.master import X\n"
        "        case _:\n"
        "            import combrum.oracle\n"
    )
    probe = tmp_path / "match_probe.py"
    probe.write_text(source)
    edges = _runtime_imports(probe, base_pkg="combrum._probe")
    assert "combrum.master" in edges
    assert "combrum.oracle" in edges


def test_walker_descends_every_compound_statement_field(tmp_path: Path) -> None:
    # The generic descent loop recurses into the "body", "orelse", "finalbody",
    # and "handlers" fields of each node, so an import buried in a ``try`` body,
    # an ``except`` handler, a ``finally`` block, or a ``for``/``while`` ``else:``
    # clause is still recorded. No real src module places a combrum import in
    # those contexts today, so test_runtime_import_graph_is_frozen never crosses
    # that boundary — yet the backend try/except guards in masters and transport
    # are exactly where a future lazy intra-package edge would land. Pin the full
    # edge set from a probe that seeds a distinct module into each field: any
    # narrowing of the descent tuple (or dropping the except-handler recursion)
    # loses at least one edge and fails the equality. Expecteds are the modules
    # the probe hand-places, not read back from the walker.
    source = (
        "def f(x):\n"
        "    try:\n"
        "        from combrum.master import A\n"  # try body
        "    except ValueError:\n"
        "        from combrum.oracle import B\n"  # except handler body
        "    finally:\n"
        "        from combrum.demand import C\n"  # finally body
        "    for _ in x:\n"
        "        pass\n"
        "    else:\n"
        "        from combrum.result import D\n"  # for ... else body
        "    while x:\n"
        "        break\n"
        "    else:\n"
        "        from combrum.parameters import E\n"  # while ... else body
    )
    probe = tmp_path / "compound_probe.py"
    probe.write_text(source)
    edges = _runtime_imports(probe, base_pkg="combrum._probe")
    assert edges == frozenset(
        {
            "combrum.master",
            "combrum.oracle",
            "combrum.demand",
            "combrum.result",
            "combrum.parameters",
        }
    )


def test_walker_excludes_type_checking_guarded_imports(tmp_path: Path) -> None:
    # The docstring makes typing-only imports (under ``if TYPE_CHECKING:``) a
    # load-bearing exclusion, but no real src module places a combrum import
    # under a TYPE_CHECKING guard, so test_runtime_import_graph_is_frozen is
    # identical whether or not the exclusion fires. Pin both directions of the
    # boundary with a probe: the guarded arm must drop out, while the runtime
    # ``else:`` sibling and a top-level import must survive. This fails if the
    # walker stops recognizing the TYPE_CHECKING guard (the excluded edge leaks
    # back into the graph and silently omits nothing — or, worse, a genuine
    # runtime edge is misclassified as typing-only and vanishes).
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from combrum.master import CutRow\n"
        "else:\n"
        "    from combrum.oracle import Oracle\n"
        "import combrum.demand\n"
    )
    probe = tmp_path / "typecheck_probe.py"
    probe.write_text(source)
    edges = _runtime_imports(probe, base_pkg="combrum._probe")
    # The typing-only arm is excluded...
    assert "combrum.master" not in edges
    # ...while the runtime ``else:`` sibling and the top-level import survive.
    assert "combrum.oracle" in edges
    assert "combrum.demand" in edges


def test_walker_excludes_attribute_form_type_checking_guard(tmp_path: Path) -> None:
    # _is_type_checking_test also matches the attribute form ``typing.TYPE_CHECKING``
    # (ast.Attribute arm), which no src module and no other test exercises. Pin it:
    # a combrum import guarded by ``if typing.TYPE_CHECKING:`` must be excluded,
    # while a sibling top-level import survives.
    source = (
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    from combrum.master import CutRow\n"
        "import combrum.demand\n"
    )
    probe = tmp_path / "typecheck_attr_probe.py"
    probe.write_text(source)
    edges = _runtime_imports(probe, base_pkg="combrum._probe")
    assert "combrum.master" not in edges
    assert "combrum.demand" in edges


# Dynamic-import callees that can couple to a combrum module by string at
# runtime while the AST import walker above sees nothing: it inspects only
# ast.Import / ast.ImportFrom, so ``importlib.import_module("combrum.result")``
# or ``__import__("combrum.result")`` never enters the frozen graph.
# find_spec is here too: ``importlib.util.find_spec("combrum.x")`` imports the
# parent package to locate the submodule, so it is a real runtime coupling.
_DYNAMIC_IMPORT_CALLEES = frozenset(
    {"import_module", "reload", "__import__", "find_spec"}
)


def _importlib_alias_bindings(tree: ast.AST) -> set[str]:
    # Local names bound to one of the dynamic-import callees via importlib, e.g.
    # ``from importlib import import_module as im`` binds ``im`` and
    # ``from importlib.util import find_spec`` binds ``find_spec``. A later
    # ``im("combrum.result")`` is the same coupling as ``import_module(...)``;
    # matching only the bare callee name would let the alias smuggle it past.
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[0] == "importlib"
        ):
            for alias in node.names:
                if alias.name in _DYNAMIC_IMPORT_CALLEES:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _combrum_string_target(arg: ast.expr) -> str | None:
    # The combrum module a dynamic-import argument names, or None. A bare literal
    # ``"combrum.result"`` yields that string; a computed argument whose leftmost
    # piece is a ``"combrum..."`` literal — ``"combrum." + tail`` or an f-string
    # ``f"combrum.{tail}"`` — yields the sentinel ``"<computed>"`` so the class of
    # runtime-assembled combrum names is not silently exempt from the gate.
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value if arg.value.split(".")[0] == "combrum" else None
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
        return "<computed>" if _combrum_string_target(arg.left) else None
    if isinstance(arg, ast.JoinedStr) and arg.values:
        head = arg.values[0]
        if (
            isinstance(head, ast.Constant)
            and isinstance(head.value, str)
            and head.value.split(".")[0] == "combrum"
        ):
            return "<computed>"
    return None


def _dynamic_combrum_couplings(path: Path) -> set[tuple[str, str]]:
    # Every dynamic-import call in ``path`` whose argument names a combrum
    # module: (callee, target). Independent of _runtime_imports — it walks Call
    # nodes, not import statements — so it witnesses exactly the couplings the
    # frozen-graph walker is blind to. Matches the callee both by bare name and
    # by any importlib alias bound in the same file, and matches the target both
    # as a literal and as a runtime-assembled ``combrum...`` string. First
    # positional or keyword argument is enough: every dynamic-import callee takes
    # the module name there.
    tree = ast.parse(path.read_text())
    aliases = _importlib_alias_bindings(tree)
    out: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            callee = func.attr
        elif isinstance(func, ast.Name):
            callee = func.id
        else:
            continue
        if callee not in _DYNAMIC_IMPORT_CALLEES and callee not in aliases:
            continue
        args = list(node.args) + [kw.value for kw in node.keywords]
        for arg in args:
            target = _combrum_string_target(arg)
            if target is not None:
                out.add((callee, target))
    return out


def test_no_dynamic_combrum_import_bypasses_frozen_graph() -> None:
    # The frozen-graph walker gates only statically-expressed imports. A coupling
    # introduced through importlib.import_module / importlib.reload / __import__ /
    # find_spec with a "combrum..." target evades it entirely, so the graph would
    # under-approximate true runtime coupling. Keep the set of such dynamic
    # couplings empty across the package: any source module that adds one must
    # be reviewed exactly like a static edge.
    found = {
        _module_name(py): couplings
        for py in SRC.rglob("*.py")
        if (couplings := _dynamic_combrum_couplings(py))
    }
    assert found == {}, (
        "dynamic intra-package coupling(s) bypass the frozen import graph: "
        f"{found} — a combrum module reached by importlib.import_module / "
        "importlib.reload / __import__ / find_spec is a runtime coupling the "
        "AST walker cannot see; express it as a static import (so the frozen "
        "graph gates it) or amend the contract explicitly"
    )


def test_dynamic_coupling_oracle_catches_aliased_and_computed_forms(
    tmp_path: Path,
) -> None:
    # The whole-package oracle above is only worth its docstring if it actually
    # sees the evasions a real regression would use. The finding: append to a frozen
    # leaf module ``def _late(): from importlib import import_module as _imp;
    # return _imp("combrum.result")`` — a real combrum.demand -> combrum.result
    # coupling that a bare-name callee match misses because the alias hides it.
    # Drive the detector directly over a probe holding every bypass form and pin
    # the exact witnessed set, so reverting any lens (alias resolution, find_spec,
    # or computed-string detection) fails here. Expecteds are hand-listed from the
    # probe, not read back from the detector.
    source = (
        "from importlib import import_module as _imp\n"
        "from importlib.util import find_spec\n"
        "import importlib\n"
        "def _aliased():\n"
        "    return _imp('combrum.result')\n"
        "def _spec():\n"
        "    return find_spec('combrum.oracle')\n"
        "def _attr_spec():\n"
        "    return importlib.util.find_spec('combrum.master')\n"
        "def _concat(tail):\n"
        "    return _imp('combrum.' + tail)\n"
        "def _fstring(tail):\n"
        "    return _imp(f'combrum.{tail}')\n"
        "def _plain():\n"
        "    return importlib.import_module('combrum.model')\n"
    )
    probe = tmp_path / "dynamic_probe.py"
    probe.write_text(source)
    couplings = _dynamic_combrum_couplings(probe)
    assert couplings == {
        ("_imp", "combrum.result"),
        ("find_spec", "combrum.oracle"),
        ("find_spec", "combrum.master"),
        ("_imp", "<computed>"),
        ("import_module", "combrum.model"),
    }


def test_dynamic_coupling_oracle_ignores_non_combrum_and_static(
    tmp_path: Path,
) -> None:
    # The oracle must not fire on the ordinary case, or the whole-package gate
    # would be a false alarm on any file that uses importlib for a stdlib/3p name
    # or that just imports combrum statically. Pin the negative: a probe with a
    # non-combrum dynamic import and a static combrum import yields no couplings.
    source = (
        "from importlib import import_module as _imp\n"
        "from combrum.demand import Demand\n"
        "import combrum.oracle\n"
        "def _f():\n"
        "    return _imp('numpy.linalg')\n"
        "def _g():\n"
        "    return __import__('json')\n"
    )
    probe = tmp_path / "dynamic_negative_probe.py"
    probe.write_text(source)
    couplings = _dynamic_combrum_couplings(probe)
    assert couplings == set()
