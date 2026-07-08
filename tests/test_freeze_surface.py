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
    # Activity-log value layer: typed events, bounded recorders, JSONL
    # sidecars. Deliberately a leaf: no engine, result, or transport imports.
    "combrum.activity": frozenset(),
    # Serial bootstrap: B cold weighted refits through build_fit_context +
    # run_fit. Weights come from the placement-invariant RNG; per-rep results
    # fold into BootstrapResult; opt-in duals stream one-in-flight to the
    # per-rep store. No solver edge: the master lives behind the builder.
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
    # Distributed-bootstrap scheduler: bounded replica waves, streamed dual
    # payloads, placement-invariant weights, and the batched reduce/exchange
    # over the transport ABC. No solver edge; observed-feature preparation
    # lives in distributed_context, and the pricing-gap certificate is reduced
    # once after all waves have priced.
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
    # Rank-agreement helpers for distributed controls, guarded rank-local
    # hooks, and warm-start theta tokens. Depends only on the transport ABC.
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
    # Adaptive-timeout callbacks. Settings apply through the SolverConfigurable
    # protocol, so no concrete solver edge; Oracle only types the hook.
    "combrum.callbacks": frozenset(
        {"combrum.oracle", "combrum.solver_settings"}
    ),
    "combrum.cut_policies": frozenset(
        {"combrum.policies", "combrum.transport.base"}
    ),
    "combrum.demand": frozenset(),
    "combrum.dual": frozenset(),
    # The either-one resolution guard. Only the transport contract (for the
    # rank-agreement round), so formulations and the oracle can consume it
    # without coupling to each other.
    "combrum.interface_resolution": frozenset({"combrum.transport.base"}),
    # The package init re-exports the estimate API and the driver / fit-step /
    # persistent-master surface, so it reaches only its own submodules.
    "combrum.engine": frozenset(
        {
            "combrum.engine.context_builder",
            "combrum.engine.driver",
            "combrum.engine.estimate",
            "combrum.engine.persistent",
        }
    ),
    # The one owner of fit-context assembly, for both the point estimate and
    # the bootstrap/sweep. Builds the FitContext, delegates observed-bundle
    # objective construction to engine.observed (which infers observed phi
    # from priced features when Model.observed_features is omitted), and
    # reaches the master only through make_master and the
    # reinstall/extract_cuts primitive.
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
    # Context assembly for the split observation/pricing axes: same envelope
    # as the dense builder, but observed feature rows come from the
    # distributed surface and only an owner-rank lazy NSlack master is built.
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
    # Observed-bundle objective materialization: infers phi rows through the
    # features either-one resolver when observed_features is omitted, and
    # reduces the objective/moment over the transport contract.
    "combrum.engine.observed": frozenset(
        {"combrum.interface_resolution", "combrum.transport.base"}
    ),
    # Gap aggregation into the Certification report: reads the gap off each
    # priced Demand and reduces counts/worst-gap across ranks.
    "combrum.engine.certify": frozenset(
        {
            "combrum.certification",
            "combrum.demand",
            "combrum.transport.base",
        }
    ),
    # The public estimate APIs: estimate uses the full-array builder,
    # estimate_distributed the split-axis one. The formulations edge is the
    # early public guard that admits NSlack only.
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
    # The driver loop composes the contracts it drives: the geometry
    # (context), the price pair it resolves once (interface_resolution), the
    # fit-step (engine.fitstep), the solve contract (formulation), the
    # schedule ABC plus the dual-concentration branch (schedule,
    # informed_schedule), the activity sink, the oracle ABC, and the
    # phase-step protocol (rowgen). No solver or master edge; it touches the
    # master only through the formulation's apply_step.
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
    # One-iteration fit-step: price / reduce+exchange / finalise / solve,
    # transport-passive over the formulation. It composes the RowGenStep
    # protocol, never a concrete formulation, master, or solver.
    "combrum.engine.fitstep": frozenset(
        {
            "combrum.demand",
            "combrum.interface_resolution",
            "combrum.rowgen",
            "combrum.transport.base",
        }
    ),
    # Persistent-master driver: one master held across an outer ψ search,
    # RHS-rewriting the carried cuts and warm-solving per ψ. Lazily defaults
    # to NSlack; the NSlack-only guard uses exact class identity, not
    # isinstance.
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
    # The row-generation methods consume frozen contracts only; solver
    # imports must stay out.
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
    # Plain dataclass module; the edges are the contracts it holds as typed
    # fields, runtime so Model annotations stay introspectable. No engine
    # edge: a value type, not a fit driver.
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
    # Run-metadata surface: carries diagnostics the run already produced,
    # never re-derives them; the runtime imports keep public result
    # annotations introspectable.
    "combrum.runinfo": frozenset(
        {
            "combrum.certification",
            "combrum.engine.driver",
            "combrum.transport.base",
        }
    ),
    # The row-generation phase contract; the engine owns the cross-rank
    # reduce/exchange, so only the demand envelope and cut row are named here.
    "combrum.rowgen": frozenset(
        {"combrum.demand", "combrum.transport.base"}
    ),
    "combrum.schedule": frozenset(),
    # The capture record the formulations emit into, so the either-one gate
    # sees every filter-chain input pre-filter. Cut-row identities are stored
    # as typed scalars, so no transport import.
    "combrum.steprecord": frozenset({"combrum._bundle_key", "combrum.demand"}),
    # SolverSettings / SolverConfigurable: framework types with no
    # concrete-solver edge.
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
    # mpi4py is loaded lazily at instantiation (an optional extra), so it
    # contributes no import edge.
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
                # so recurse explicitly or an import inside a case arm would
                # vanish from the graph.
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
        f"gone={sorted(set(FROZEN_IMPORTS) - set(actual))}; "
        "amend FROZEN_IMPORTS explicitly"
    )
    for module, expected in FROZEN_IMPORTS.items():
        got = actual[module]
        assert got == expected, (
            f"{module}: runtime imports changed — "
            f"added={sorted(got - expected)}, removed={sorted(expected - got)}"
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
    # src uses only absolute imports, so the ``node.level > 0`` branch of
    # _from_edges (and _resolve_relative behind it) is never hit by the
    # frozen-graph test. Check it directly against Python's relative-import
    # rules: strip ``level - 1`` trailing components of the importing package,
    # then attach the module tail (or each bare name).
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
    # Multi-level: ``from ..`` inside combrum.engine strips one component back
    # to combrum, so a bare name resolves to combrum.x.
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
    # An import buried in a ``try`` body, an ``except`` handler, a ``finally``
    # block, or a ``for``/``while`` ``else:`` clause must still be recorded —
    # the backend try/except guards in masters and transport are where a lazy
    # intra-package edge would land.
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
    # The ``if TYPE_CHECKING:`` arm must drop out, while the runtime ``else:``
    # sibling and a top-level import stay.
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
    assert "combrum.master" not in edges
    assert "combrum.oracle" in edges
    assert "combrum.demand" in edges


def test_walker_excludes_attribute_form_type_checking_guard(tmp_path: Path) -> None:
    # _is_type_checking_test also matches the attribute form
    # ``typing.TYPE_CHECKING``.
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


# Callees that import a combrum module by string, invisible to the AST import
# walker above (it inspects only ast.Import / ast.ImportFrom). find_spec
# counts: ``importlib.util.find_spec("combrum.x")`` imports the parent package
# to locate the submodule.
_DYNAMIC_IMPORT_CALLEES = frozenset(
    {"import_module", "reload", "__import__", "find_spec"}
)


def _importlib_alias_bindings(tree: ast.AST) -> set[str]:
    # Local names bound to one of the dynamic-import callees via importlib,
    # e.g. ``from importlib import import_module as im`` binds ``im``. A later
    # ``im("combrum.result")`` is the same coupling as ``import_module(...)``.
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
    # The combrum module a dynamic-import argument names, or None. A computed
    # argument whose leftmost piece is a ``"combrum..."`` literal —
    # ``"combrum." + tail`` or ``f"combrum.{tail}"`` — yields the sentinel
    # ``"<computed>"`` so runtime-assembled names still register.
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
    # module, as (callee, target) pairs. Matches the callee by bare name or
    # importlib alias, and the target as a literal or runtime-assembled
    # ``combrum...`` string. Checking every positional or keyword argument is
    # enough: each dynamic-import callee takes the module name there.
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
    # The frozen graph gates only static imports; keep the package free of
    # dynamic combrum imports so the graph stays a complete picture of runtime
    # coupling.
    found = {
        _module_name(py): couplings
        for py in SRC.rglob("*.py")
        if (couplings := _dynamic_combrum_couplings(py))
    }
    assert found == {}, (
        "dynamic intra-package coupling(s) bypass the frozen import graph: "
        f"{found}; express as a static import so the graph gates it"
    )


def test_dynamic_coupling_scan_catches_aliased_and_computed_forms(
    tmp_path: Path,
) -> None:
    # One probe per bypass form: importlib alias, find_spec (bare and
    # attribute), string concatenation, f-string, and the plain attribute call.
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


def test_dynamic_coupling_scan_ignores_non_combrum_and_static(
    tmp_path: Path,
) -> None:
    # Non-combrum dynamic imports and static combrum imports must not register.
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
