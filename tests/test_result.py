from __future__ import annotations

import copy
import json
import pickle
import warnings

import numpy as np
import pytest

from combrum.parameters import Parameters
from combrum.result import BootstrapResult, FitResult


def make_params() -> Parameters:
    return Parameters({"beta": (-5.0, 5.0, 2), "gamma": (0.0, 10.0, 1)})


def make_fit(
    theta: tuple[float, ...] = (1.0, 2.0, 3.0), **overrides: object
) -> FitResult:
    kwargs: dict[str, object] = dict(
        theta_hat=np.array(theta),
        objective=0.5,
        empirical_moment=np.array([0.1, 0.2, 0.3]),
        runtime_seconds=1.5,
        n_active_cuts=4,
        parameters=make_params(),
    )
    kwargs.update(overrides)
    return FitResult(**kwargs)  # type: ignore[arg-type]


def _percentile_linear(sorted_col: np.ndarray, tail: float) -> float:
    # Independent plain-Python linear-interpolation percentile (numpy 'linear'
    # method), so ci()'s level->tail mapping is checked against a distinct oracle.
    n = len(sorted_col)
    pos = tail / 100.0 * (n - 1)
    lo_i = int(np.floor(pos))
    frac = pos - lo_i
    hi_i = min(lo_i + 1, n - 1)
    return float(sorted_col[lo_i] + frac * (sorted_col[hi_i] - sorted_col[lo_i]))


def _ci_oracle(thetas: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    tail = 100.0 * (1.0 - level) / 2.0
    K = thetas.shape[1]
    lo = np.empty(K)
    hi = np.empty(K)
    for k in range(K):
        col = np.sort(thetas[:, k])
        lo[k] = _percentile_linear(col, tail)
        hi[k] = _percentile_linear(col, 100.0 - tail)
    return lo, hi


def _cov_oracle(rows: np.ndarray) -> np.ndarray:
    # Plain-Python ddof=1 covariance; structurally distinct from np.cov so a
    # ddof/rowvar regression in cov() is caught by value.
    B, K = rows.shape
    means = [sum(rows[b, k] for b in range(B)) / B for k in range(K)]
    cov = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            acc = 0.0
            for b in range(B):
                acc += (rows[b, i] - means[i]) * (rows[b, j] - means[j])
            cov[i, j] = acc / (B - 1)
    return cov


def _se_oracle(rows: np.ndarray) -> np.ndarray:
    # Plain-Python ddof=1 standard error, distinct from np.std(ddof=1), so an
    # se() that summed over the wrong rows (or ddof) is caught by value.
    B, K = rows.shape
    out = np.empty(K)
    for k in range(K):
        mean = sum(rows[b, k] for b in range(B)) / B
        acc = sum((rows[b, k] - mean) ** 2 for b in range(B))
        out[k] = (acc / (B - 1)) ** 0.5
    return out


def make_boot(**overrides: object) -> BootstrapResult:
    kwargs: dict[str, object] = dict(
        thetas=np.array(
            [
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [9.0, 9.0, 9.0],
                [0.9, 1.9, 2.9],
            ]
        ),
        converged=np.array([True, True, False, True]),
        parameters=make_params(),
    )
    kwargs.update(overrides)
    return BootstrapResult(**kwargs)  # type: ignore[arg-type]


def test_fit_result_named_accessors() -> None:
    fit = make_fit()
    named = fit.theta_named()
    np.testing.assert_array_equal(named["beta"], [1.0, 2.0])
    np.testing.assert_array_equal(named["gamma"], [3.0])
    moments = fit.empirical_moment_named()
    np.testing.assert_array_equal(moments["beta"], [0.1, 0.2])
    np.testing.assert_array_equal(moments["gamma"], [0.3])


def test_fit_result_validation() -> None:
    with pytest.raises(ValueError, match=r"theta_hat must have shape \(K,\)"):
        make_fit(theta=(1.0, 2.0))
    with pytest.raises(ValueError, match="empirical_moment must have shape"):
        make_fit(empirical_moment=np.zeros(4))
    with pytest.raises(ValueError, match="n_active_cuts must be >= 0"):
        make_fit(n_active_cuts=-1)
    # Zero is the inclusive lower boundary the message promises is legal: a
    # `<= 0` guard would wrongly reject a fit that activated no cuts.
    assert make_fit(n_active_cuts=0).n_active_cuts == 0
    # Non-finite estimates must be rejected at construction; otherwise a NaN/inf
    # theta_hat or moment is stored and contaminates every downstream summary.
    with pytest.raises(ValueError, match="theta_hat must be finite"):
        make_fit(theta=(np.nan, 2.0, 3.0))
    with pytest.raises(ValueError, match="empirical_moment must be finite"):
        make_fit(empirical_moment=np.array([np.inf, 0.2, 0.3]))


def test_fit_result_to_dict_json_round_trip() -> None:
    fit = make_fit(slack=np.array([0.0, 0.5]), metadata={"seed": 7})
    doc = fit.to_dict()
    # Contract: to_dict exposes exactly these JSON-ready fields and excludes the
    # run_info/cuts/cut_duals provenance, so a leaked extra key (even a
    # serializable one) is a regression.
    assert set(doc) == {
        "theta_hat",
        "objective",
        "empirical_moment",
        "runtime_seconds",
        "n_active_cuts",
        "slack",
        "metadata",
    }
    restored = json.loads(json.dumps(doc))
    assert restored == doc
    assert restored["theta_hat"] == [1.0, 2.0, 3.0]
    assert restored["objective"] == 0.5
    assert restored["empirical_moment"] == [0.1, 0.2, 0.3]
    assert restored["runtime_seconds"] == 1.5
    assert restored["n_active_cuts"] == 4
    assert restored["slack"] == [0.0, 0.5]
    assert restored["metadata"] == {"seed": 7}


def test_slack_summary_requires_slack() -> None:
    with pytest.raises(ValueError, match="requires the slack field"):
        make_fit().slack_summary()
    # An empty slack vector is a distinct error from slack=None: without the
    # guard, mean_slack is nan and max_slack raises a cryptic numpy message.
    with pytest.raises(ValueError, match="nonempty slack vector"):
        make_fit(slack=np.array([])).slack_summary()


def test_slack_summary_from_slack() -> None:
    # Asymmetric slack: 3 zeros vs 2 nonzeros, so n_binding=3 distinguishes the
    # zero-slack count from counting nonzero (=2) or >0 (=2) slack.
    fit = make_fit(slack=np.array([0.0, 0.0, 0.0, 1.0, 3.0]))
    summary = fit.slack_summary()
    assert summary["total_slack"] == 4.0
    assert summary["mean_slack"] == 0.8
    assert summary["max_slack"] == 3.0
    assert summary["n_binding"] == 3

    # n_binding counts *exactly* zero slack, not slack below a tolerance. Put one
    # entry strictly inside (0, 1e-9): it is nonbinding, so n_binding stays 3,
    # but a count_nonzero(slack < 1e-9) regression would report 4.
    tiny = np.array([0.0, 0.0, 0.0, 1e-12, 1.0, 3.0])
    tiny_summary = make_fit(slack=tiny).slack_summary()
    # Independent plain-Python oracle for the whole summary.
    vals = tiny.tolist()
    exp_total = 0.0
    for v in vals:
        exp_total += v
    exp_binding = sum(1 for v in vals if v == 0.0)
    exp_max = vals[0]
    for v in vals[1:]:
        if v > exp_max:
            exp_max = v
    assert tiny_summary["total_slack"] == pytest.approx(exp_total)
    assert tiny_summary["mean_slack"] == pytest.approx(exp_total / len(vals))
    assert tiny_summary["max_slack"] == exp_max
    assert tiny_summary["n_binding"] == exp_binding == 3


def test_fit_result_arrays_read_only() -> None:
    fit = make_fit(slack=np.array([0.0, 0.5]))
    assert fit.slack is not None
    for arr in (fit.theta_hat, fit.empirical_moment, fit.slack):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr[0] = 99.0


def test_bootstrap_result_arrays_read_only() -> None:
    boot = make_boot(u_samples=np.zeros((4, 5)))
    assert boot.u_samples is not None
    for arr in (boot.thetas, boot.converged, boot.u_samples):
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flat[0] = 99.0


def test_fit_and_bootstrap_results_pickle_and_deepcopy_round_trip() -> None:
    fit = make_fit(slack=np.array([0.0, 0.5]), metadata={"seed": 7})
    boot = make_boot(point_estimate=fit, metadata={"kind": "bootstrap"})

    for source in (fit, boot):
        for restored in (
            copy.deepcopy(source),
            pickle.loads(pickle.dumps(source)),
        ):
            assert restored.parameters == source.parameters
            assert restored.metadata == source.metadata
            if isinstance(source, FitResult):
                np.testing.assert_array_equal(restored.theta_hat, source.theta_hat)
                np.testing.assert_array_equal(
                    restored.empirical_moment, source.empirical_moment
                )
                # slack must survive the round trip (a __getstate__ that dropped
                # it would leave restored.slack=None).
                assert restored.slack is not None
                np.testing.assert_array_equal(restored.slack, source.slack)
                assert restored.objective == source.objective
                assert restored.runtime_seconds == source.runtime_seconds
                assert restored.n_active_cuts == source.n_active_cuts
                # The read-only invariant must survive the round trip: numpy
                # drops the WRITEABLE flag through pickle/deepcopy, so
                # __setstate__ re-freezes the arrays. Without it these come back
                # mutable.
                for arr in (
                    restored.theta_hat,
                    restored.empirical_moment,
                    restored.slack,
                ):
                    assert not arr.flags.writeable
            else:
                np.testing.assert_array_equal(restored.thetas, source.thetas)
                np.testing.assert_array_equal(restored.converged, source.converged)
                # The read-only invariant must survive the round trip (see the
                # FitResult branch): __setstate__ re-freezes the arrays.
                assert not restored.thetas.flags.writeable
                assert not restored.converged.flags.writeable
                # point_estimate provenance must survive; a payload that dropped
                # it would restore point_estimate=None.
                assert restored.point_estimate is not None
                np.testing.assert_array_equal(
                    restored.point_estimate.theta_hat, source.point_estimate.theta_hat
                )
                np.testing.assert_array_equal(
                    restored.point_estimate.slack, source.point_estimate.slack
                )


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match=r"thetas must have shape \(B, K\)"):
        make_boot(thetas=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="B must be >= 1"):
        make_boot(thetas=np.zeros((0, 3)), converged=np.zeros(0, dtype=bool))
    # One replication is the inclusive lower boundary: a `B < 2` guard would
    # wrongly reject a single-replication bootstrap.
    single = make_boot(thetas=np.zeros((1, 3)), converged=np.array([True]))
    assert single.thetas.shape == (1, 3)
    assert single.converged.shape == (1,)
    assert single.n_converged == 1
    with pytest.raises(ValueError, match=r"converged must have shape \(B,\)"):
        make_boot(converged=np.array([True, False]))
    with pytest.raises(ValueError, match="one payload per replication"):
        make_boot(duals=("d0", "d1"))
    with pytest.raises(ValueError, match="leading dimension B"):
        make_boot(u_samples=np.zeros((3, 5)))
    with pytest.raises(ValueError, match="thetas must be finite"):
        make_boot(
            thetas=np.array([[np.nan, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        )


def test_bootstrap_selection_semantics() -> None:
    boot = make_boot()
    assert boot.n_converged == 3
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        np.testing.assert_allclose(boot.mean(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        boot.mean(only_converged=False), boot.thetas.mean(axis=0)
    )
    none_converged = make_boot(converged=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="at least one converged replication"):
        none_converged.mean()
    np.testing.assert_allclose(
        none_converged.mean(only_converged=False),
        none_converged.thetas.mean(axis=0),
    )


def test_bootstrap_ddof1_summaries_require_two_selected_replications() -> None:
    one = make_boot(
        thetas=np.array([[1.0, 2.0, 3.0]]),
        converged=np.array([True]),
    )

    with pytest.raises(ValueError, match="se requires at least two"):
        one.se()
    with pytest.raises(ValueError, match="cov requires at least two"):
        one.cov()


def test_bootstrap_exclusion_warning_semantics() -> None:
    # Every default summary over a partially-converged result warns; the
    # all-converged and include-all paths stay silent.
    partial = make_boot()
    # Pin summaries against the selected rows, not just the warning.
    selected = partial.thetas[partial.converged]
    assert selected.shape == (3, 3)
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        np.testing.assert_allclose(partial.se(), _se_oracle(selected))
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        np.testing.assert_allclose(partial.cov(), _cov_oracle(selected))
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        lo, hi = partial.ci(level=0.9)
    exp_lo, exp_hi = _ci_oracle(selected, 0.9)
    np.testing.assert_allclose(lo, exp_lo)
    np.testing.assert_allclose(hi, exp_hi)
    # The masked values must differ from the all-row summaries; otherwise the
    # oracle above would not distinguish selection from no selection.
    assert not np.allclose(partial.se(only_converged=False), _se_oracle(selected))

    all_ok = make_boot(converged=np.ones(4, dtype=bool))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        all_ok.mean()
        all_ok.se()
        partial.mean(only_converged=False)


def test_bootstrap_summary_shapes() -> None:
    boot = make_boot(converged=np.ones(4, dtype=bool))
    selected = boot.thetas[boot.converged]
    assert boot.se().shape == (3,)
    np.testing.assert_allclose(boot.se(), selected.std(axis=0, ddof=1))
    assert boot.cov().shape == (3, 3)
    np.testing.assert_allclose(boot.cov(), _cov_oracle(boot.thetas))
    lo, hi = boot.ci(level=0.9)
    assert lo.shape == (3,) and hi.shape == (3,)
    assert np.all(lo <= hi)
    # ci() must honour level: the band matches the 5/95 percentiles for 0.9 and
    # differs from the 25/75 band for 0.5, so a level-ignoring tail fails.
    exp_lo, exp_hi = _ci_oracle(boot.thetas, 0.9)
    np.testing.assert_allclose(lo, exp_lo)
    np.testing.assert_allclose(hi, exp_hi)
    assert not np.allclose(lo, boot.ci(level=0.5)[0])
    with pytest.raises(ValueError, match=r"level must lie in \(0, 1\)"):
        boot.ci(level=1.0)


def test_bootstrap_named_summaries() -> None:
    boot = make_boot(converged=np.ones(4, dtype=bool))
    se_named = boot.se_named()
    assert set(se_named) == {"beta", "gamma"}
    assert se_named["beta"].shape == (2,)
    assert se_named["gamma"].shape == (1,)
    se_flat = boot.se()
    np.testing.assert_array_equal(se_named["beta"], se_flat[:2])
    np.testing.assert_array_equal(se_named["gamma"], se_flat[2:])
    ci_named = boot.ci_named(level=0.9)
    ci_lo, ci_hi = boot.ci(level=0.9)
    lo_beta, hi_beta = ci_named["beta"]
    assert lo_beta.shape == (2,) and hi_beta.shape == (2,)
    np.testing.assert_array_equal(lo_beta, ci_lo[:2])
    np.testing.assert_array_equal(hi_beta, ci_hi[:2])
    lo_gamma, hi_gamma = ci_named["gamma"]
    assert lo_gamma.shape == (1,) and hi_gamma.shape == (1,)
    np.testing.assert_array_equal(lo_gamma, ci_lo[2:])
    np.testing.assert_array_equal(hi_gamma, ci_hi[2:])


def test_named_summaries_forward_only_converged() -> None:
    # The named methods must forward only_converged through the block split, not
    # silently default to converged-only. Use the partially-converged default
    # fixture (the [9, 9, 9] outlier row is non-converged) so the all-row and
    # converged-only summaries are far apart, and pin the full named dict against
    # the independent oracle over *every* row.
    partial = make_boot()
    all_rows = partial.thetas
    assert all_rows.shape == (4, 3)

    se_all = _se_oracle(all_rows)
    se_named_all = partial.se_named(only_converged=False)
    assert set(se_named_all) == {"beta", "gamma"}
    np.testing.assert_allclose(se_named_all["beta"], se_all[:2])
    np.testing.assert_allclose(se_named_all["gamma"], se_all[2:])

    lo_all, hi_all = _ci_oracle(all_rows, 0.9)
    ci_named_all = partial.ci_named(level=0.9, only_converged=False)
    assert set(ci_named_all) == {"beta", "gamma"}
    lo_beta, hi_beta = ci_named_all["beta"]
    lo_gamma, hi_gamma = ci_named_all["gamma"]
    np.testing.assert_allclose(lo_beta, lo_all[:2])
    np.testing.assert_allclose(hi_beta, hi_all[:2])
    np.testing.assert_allclose(lo_gamma, lo_all[2:])
    np.testing.assert_allclose(hi_gamma, hi_all[2:])

    # The converged-only named summaries must genuinely differ, so the oracle
    # above distinguishes forwarded selection from a hardcoded only_converged.
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        se_named_conv = partial.se_named()
    assert not np.allclose(se_named_conv["beta"], se_all[:2])
    with pytest.warns(UserWarning, match="exclude 1 non-converged"):
        ci_named_conv = partial.ci_named(level=0.9)
    assert not np.allclose(ci_named_conv["beta"][0], lo_all[:2])


def test_concat_merges_replications() -> None:
    # concat keeps the first shard's point-estimate provenance.
    first_point = make_fit(slack=np.array([0.0, 0.5]), metadata={"shard": "first"})
    second_point = make_fit(slack=np.array([9.0, 9.0]), metadata={"shard": "second"})
    assert first_point is not second_point
    np.testing.assert_array_equal(first_point.theta_hat, second_point.theta_hat)
    first = make_boot(
        thetas=np.ones((2, 3)),
        converged=np.array([True, False]),
        point_estimate=first_point,
        u_samples=np.zeros((2, 5)),
        duals=("d0", "d1"),
    )
    second = make_boot(
        thetas=2.0 * np.ones((3, 3)),
        converged=np.array([True, True, True]),
        point_estimate=second_point,
        u_samples=np.ones((3, 5)),
        duals=("d2", "d3", "d4"),
    )
    merged = BootstrapResult.concat([first, second])
    assert merged.thetas.shape == (5, 3)
    np.testing.assert_array_equal(merged.thetas[:2], first.thetas)
    np.testing.assert_array_equal(merged.thetas[2:], second.thetas)
    np.testing.assert_array_equal(
        merged.converged, [True, False, True, True, True]
    )
    # First-shard provenance, by identity and by the distinguishing fields.
    assert merged.point_estimate is first_point
    np.testing.assert_array_equal(
        merged.point_estimate.slack, first_point.slack
    )
    assert merged.point_estimate.metadata == {"shard": "first"}
    assert merged.u_samples is not None
    assert merged.u_samples.shape == (5, 5)
    # The u_samples payload must stay row-aligned with thetas/converged/duals:
    # the first shard's all-zero rows lead, the second shard's all-one rows
    # follow. A reversed/misaligned concat keeps shape (5, 5) but fails here.
    np.testing.assert_array_equal(merged.u_samples[:2], first.u_samples)
    np.testing.assert_array_equal(merged.u_samples[2:], second.u_samples)
    assert merged.duals == ("d0", "d1", "d2", "d3", "d4")


def test_concat_aggregates_certification_metadata() -> None:
    # Two distinct nonzero finite worst gaps: merged worst_gap must be the max
    # (7.0), not the sum (10.0), mean (5.0), or a single shard's value.
    #
    # Each shard also carries non-certification metadata so concat's generic
    # merge loop is exercised: a shard-local key must survive, and a shared key
    # ("tag") must resolve to the *later* shard's value.
    low_cert = {
        "n_priced": 4,
        "n_inexact": 1,
        "worst_gap": 3.0,
        "worst_gap_unknown": False,
    }
    high_cert = {
        "n_priced": 6,
        "n_inexact": 2,
        "worst_gap": 7.0,
        "worst_gap_unknown": False,
    }
    low_gap = make_boot(
        metadata={"certification": low_cert, "seed": 1, "tag": "low"}
    )
    high_gap = make_boot(
        metadata={"certification": high_cert, "note": "x", "tag": "high"}
    )

    merged_cert = {
        "n_priced": 10,
        "n_inexact": 3,
        "worst_gap": 7.0,
        "worst_gap_unknown": False,
    }
    # Full-metadata oracle for each shard order: aggregated certification plus
    # both shard-local keys, with the later shard's "tag" winning the collision.
    expected_by_order = {
        (id(low_gap), id(high_gap)): {
            "seed": 1,
            "note": "x",
            "tag": "high",
            "certification": merged_cert,
        },
        (id(high_gap), id(low_gap)): {
            "seed": 1,
            "note": "x",
            "tag": "low",
            "certification": merged_cert,
        },
    }
    for shards in ([low_gap, high_gap], [high_gap, low_gap]):
        merged = BootstrapResult.concat(shards)
        assert merged.metadata == expected_by_order[(id(shards[0]), id(shards[1]))]


def test_concat_aggregates_unknown_certification_gap() -> None:
    unknown = make_boot(
        metadata={
            "certification": {
                "n_priced": 5,
                "n_inexact": 1,
                "worst_gap": None,
                "worst_gap_unknown": True,
            }
        }
    )
    finite = make_boot(
        metadata={
            "certification": {
                "n_priced": 7,
                "n_inexact": 1,
                "worst_gap": 2.0,
                "worst_gap_unknown": False,
            }
        }
    )

    merged = BootstrapResult.concat([finite, unknown])

    assert merged.metadata["certification"] == {
        "n_priced": 12,
        "n_inexact": 2,
        "worst_gap": None,
        "worst_gap_unknown": True,
    }


def test_concat_rejects_inconsistent_provenance_and_payloads() -> None:
    # Payloads are all-or-none across shards; a partial set is rejected.
    with_payload = make_boot(u_samples=np.zeros((4, 5)), duals=tuple("abcd"))
    bare = make_boot()
    with pytest.raises(ValueError, match="u_samples on all shards or on none"):
        BootstrapResult.concat([with_payload, bare])
    with pytest.raises(ValueError, match="duals on all shards or on none"):
        BootstrapResult.concat([make_boot(duals=tuple("abcd")), bare])
    with pytest.raises(ValueError, match="certification metadata"):
        BootstrapResult.concat(
            [
                make_boot(
                    metadata={
                        "certification": {
                            "n_priced": 1,
                            "n_inexact": 0,
                            "worst_gap": 0.0,
                            "worst_gap_unknown": False,
                        }
                    }
                ),
                bare,
            ]
        )
    # Ragged payload shapes across shards must be rejected, not force-aligned.
    with pytest.raises(ValueError, match="matching u_samples payload shapes"):
        BootstrapResult.concat(
            [
                make_boot(u_samples=np.zeros((4, 5))),
                make_boot(u_samples=np.zeros((4, 7))),
            ]
        )
    # Point-estimate provenance: all-or-none, and identical when present.
    point = make_fit()
    with pytest.raises(ValueError, match="provenance on all shards or on none"):
        BootstrapResult.concat([make_boot(point_estimate=point), bare])
    other_point = make_fit(theta_hat=np.array([9.0, 9.0, 9.0]))
    with pytest.raises(ValueError, match="identical point_estimate"):
        BootstrapResult.concat(
            [
                make_boot(point_estimate=point),
                make_boot(point_estimate=other_point),
            ]
        )


def test_concat_rejects_layout_mismatch() -> None:
    other_layout = Parameters({"beta": (-5.0, 5.0, 3)})
    other = BootstrapResult(
        thetas=np.zeros((2, 3)),
        converged=np.array([True, True]),
        parameters=other_layout,
    )
    with pytest.raises(ValueError, match="identical parameter layouts"):
        BootstrapResult.concat([make_boot(), other])
    with pytest.raises(ValueError, match="at least one BootstrapResult"):
        BootstrapResult.concat([])
