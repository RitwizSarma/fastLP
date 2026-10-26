"""Manual-equivalence and reporting checks for structured fit terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastlp import LocalProjection, sugar as s


def moderation_panel() -> pd.DataFrame:
    rng = np.random.default_rng(9271)
    units, periods = 30, 12
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(units), periods),
        "time": np.tile(np.arange(periods), units),
    })
    data["location"] = data["unit"] % 5
    data["treated"] = (data["unit"] // 5 % 2).astype(float)
    data["group"] = ((data["location"] + data["time"]) % 3).astype(str)
    weather = rng.normal(size=(5, periods))
    data["event"] = weather[data["location"], data["time"]]
    data["y"] = (
        0.3 * data["event"] * data["treated"]
        + 0.1 * data["treated"] * (data["group"] == "1")
        + rng.normal(size=len(data))
    )
    return data


@pytest.mark.parametrize("covariance", ["hc0", "cluster"])
def test_categorical_shock_and_joint_fe_match_manual_design(covariance: str) -> None:
    data = moderation_panel()
    data["loc_time"] = pd.factorize(pd.MultiIndex.from_frame(data[["location", "time"]]))[0]
    for level in ("0", "1", "2"):
        data[f"shock_{level}"] = data["event"] * data["treated"] * (data["group"] == level)
    for level in ("1", "2"):
        data[f"treated_{level}"] = data["treated"] * (data["group"] == level)
    options = dict(outcome="y", unit="unit", time="time")
    if covariance == "cluster":
        options["cluster"] = "unit"
    actual = LocalProjection(horizons=2, covariance=covariance, few_cluster_threshold=None).fit(
        data,
        shock=s.interact("event", "treated", by=s.C("group", levels=["0", "1", "2"]), name="event_by_group"),
        fixed_effects=[s.fe("location", "time"), "unit"],
        **options,
    )
    expected = LocalProjection(horizons=2, covariance=covariance, few_cluster_threshold=None).fit(
        data,
        shock=["shock_0", "shock_1", "shock_2"],
        controls=["treated_1", "treated_2"],
        fixed_effects=["loc_time", "unit"],
        **options,
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-10)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_, atol=1e-10)
    responses = actual.irf_frame("event_by_group")
    assert responses["level"].unique().tolist() == ["0", "1", "2"]
    assert responses["level_n_obs"].gt(0).all()
    assert any(item["term"] == "event" for item in actual.absorbed_terms_)
    ax = actual.plot_irf("event_by_group")
    assert len(ax.lines) == 4  # Three responses and the zero line.
    assert len(ax.collections) == 3
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_continuous_moderation_uses_full_covariance() -> None:
    rng = np.random.default_rng(19)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(20), 10),
        "time": np.tile(np.arange(10), 20),
        "x": rng.normal(size=200),
        "z": rng.normal(size=200),
        "y": rng.normal(size=200),
    })
    data["xz"] = data["x"] * data["z"]
    actual = LocalProjection(horizons=1, covariance="hc1").fit(
        data, outcome="y", shock=s.interact("x", by="z", name="conditional_x"),
        unit="unit", time="time",
    )
    expected = LocalProjection(horizons=1, covariance="hc1").fit(
        data, outcome="y", shock=["x", "xz"], controls=["z"],
        unit="unit", time="time",
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_)
    result = actual.irf_frame("conditional_x", at=[0.0, 2.0])
    for horizon in range(2):
        row = result.loc[(result.horizon == horizon) & (result["at"] == 2.0)].iloc[0]
        contrast = np.array([0.0, 1.0, 2.0, 0.0])
        np.testing.assert_allclose(row.estimate, contrast @ actual.coef_[horizon])
        np.testing.assert_allclose(
            row.std_error,
            np.sqrt(contrast @ actual.covariance_[horizon] @ contrast),
        )
    ax = actual.plot_irf("conditional_x", at=[0.0, 2.0])
    assert len(ax.collections) == 2
    import matplotlib.pyplot as plt

    plt.close(ax.figure)


def test_five_way_control_and_completed_shock_lag_match_manual() -> None:
    rng = np.random.default_rng(77)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(20), 8),
        "time": np.tile(np.arange(8), 20),
        "y": rng.normal(size=160),
    })
    for name in "abcdef":
        data[name] = rng.normal(size=len(data))
    data["ab"] = data.a * data.b
    data["abcde"] = data.a * data.b * data.c * data.d * data.e
    data["abcdef"] = data["abcde"] * data.f
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data, outcome="y", shock=s.interact("a", "b", name="ab"),
        controls=[
            s.interact("a", "b", "c", "d", "e"),
            s.interact("a", "b", "c", "d", "e", "f"),
        ],
        shock_lags=1, unit="unit", time="time",
    )
    expected = LocalProjection(horizons=1, covariance="hc0").fit(
        data, outcome="y", shock="ab",
        controls=["a", "b", "abcde", "abcdef"], shock_lags=1,
        unit="unit", time="time",
    )
    for actual_name, expected_name in (
        ("Intercept", "Intercept"), ("a:b", "ab"), ("a:b:c:d:e", "abcde"),
        ("a:b:c:d:e:f", "abcdef"),
        ("a", "a"), ("b", "b"), ("a:b_lag1", "ab_lag1"),
    ):
        left = list(actual.feature_names_in_).index(actual_name)
        right = list(expected.feature_names_in_).index(expected_name)
        np.testing.assert_allclose(actual.coef_[:, left], expected.coef_[:, right])


def test_five_key_fixed_effect_matches_precomputed_id() -> None:
    rng = np.random.default_rng(208)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(18), 8),
        "time": np.tile(np.arange(8), 18),
        "x": rng.normal(size=144),
        "y": rng.normal(size=144),
    })
    for name, modulus in zip("abcde", (2, 3, 2, 3, 2), strict=True):
        data[name] = data.unit % modulus
    data["joint"] = pd.factorize(pd.MultiIndex.from_frame(data[list("abcde")]))[0]
    options = dict(outcome="y", shock="x", unit="unit", time="time")
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data, fixed_effects=[s.fe(*"abcde")], **options
    )
    expected = LocalProjection(horizons=1, covariance="hc0").fit(
        data, fixed_effects=["joint"], **options
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_)


def test_absorbed_requested_shock_level_raises() -> None:
    data = moderation_panel()
    with pytest.raises(ValueError, match="negligible within variation"):
        LocalProjection(horizons=0, covariance="hc0").fit(
            data, outcome="y",
            shock=s.interact("event", by=s.C("group", levels=["0", "1", "2"])),
            unit="unit", time="time", fixed_effects=[s.fe("location", "time")],
        )


def test_progress_callback_and_default_silence(capsys) -> None:
    data = moderation_panel()
    events = []
    options = dict(outcome="y", shock="event", unit="unit", time="time")
    quiet = LocalProjection(horizons=1, covariance="hc0").fit(data, **options)
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data, progress_callback=events.append, **options
    )
    np.testing.assert_allclose(actual.coef_, quiet.coef_)
    assert capsys.readouterr().err == ""
    assert events[0].stage == "validate specification"
    assert any(event.stage == "estimate horizon batch" and event.completed == 2 for event in events)
    assert all(event.elapsed_seconds >= 0 for event in events)


def test_progress_text_and_exception_cleanup(capsys) -> None:
    data = moderation_panel()
    LocalProjection(horizons=0, covariance="hc0").fit(
        data, outcome="y", shock="event", unit="unit", time="time", progress=True
    )
    output = capsys.readouterr().err
    assert "prepare panel" in output
    assert "fit complete" in output
    with pytest.raises(ValueError, match="missing required columns"):
        LocalProjection(horizons=0, covariance="hc0").fit(
            data, outcome="y", shock="absent", unit="unit", time="time", progress=True
        )
    assert "fit failed" in capsys.readouterr().err


def test_polars_sugar_matches_pandas() -> None:
    pl = pytest.importorskip("polars")
    data = moderation_panel()
    polars_data = pl.DataFrame({name: data[name].to_numpy() for name in data.columns})
    options = dict(
        outcome="y",
        shock=s.interact("event", "treated", by=s.C("group", levels=["0", "1", "2"])),
        shock_lags=1,
        unit="unit", time="time",
        fixed_effects=[s.fe("location", "time"), "unit"],
    )
    expected = LocalProjection(horizons=1, covariance="hc0").fit(data, **options)
    actual = LocalProjection(horizons=1, covariance="hc0").fit(polars_data.lazy(), **options)
    np.testing.assert_allclose(actual.coef_, expected.coef_, atol=1e-10)
    np.testing.assert_allclose(actual.covariance_, expected.covariance_, atol=1e-10)


def test_sugar_memory_budget_rejects_before_panel_preparation(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("panel preparation should not start")

    monkeypatch.setattr("fastlp.estimator.prepare_panel", forbidden)
    with pytest.raises(ValueError, match="memory_budget=.*planned minimum"):
        LocalProjection(horizons=2, covariance="hc0", memory_budget=1024).fit(
            moderation_panel(), outcome="y",
            shock=s.interact(
                "event", "treated", by=s.C("group", levels=["0", "1", "2"])
            ),
            unit="unit", time="time",
        )


def test_named_control_lag_maps_to_completed_product() -> None:
    rng = np.random.default_rng(881)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(12), 7),
        "time": np.tile(np.arange(7), 12),
        "y": rng.normal(size=84),
        "x": rng.normal(size=84),
        "a": rng.normal(size=84),
        "b": rng.normal(size=84),
    })
    data["ab"] = data.a * data.b
    options = dict(outcome="y", shock="x", unit="unit", time="time")
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data, controls=[s.interact("a", "b", name="ab_term")],
        control_lags={"ab_term": 1}, **options
    )
    expected = LocalProjection(horizons=1, covariance="hc0").fit(
        data, controls=["ab"], control_lags=1, **options
    )
    np.testing.assert_allclose(actual.coef_, expected.coef_)


def test_control_lag_uses_deduplicated_product_name() -> None:
    rng = np.random.default_rng(312)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(20), 8),
        "time": np.tile(np.arange(8), 20),
        **{name: rng.normal(size=160) for name in ("y", "a", "b", "c")},
    })
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data, outcome="y", shock=s.interact("a", "b", "c"),
        controls=[s.interact("b", "a", name="ba")],
        control_lags={"ba": 1}, unit="unit", time="time",
    )
    assert "a:b_lag1" in actual.feature_names_in_


def test_duplicate_control_lag_alias_is_rejected() -> None:
    data = moderation_panel()
    with pytest.raises(ValueError, match="ambiguous control_lags key 'same'"):
        LocalProjection(horizons=1, covariance="hc0").fit(
            data, outcome="y", shock="event",
            controls=[
                s.interact("event", "treated", name="same"),
                s.interact("event", "location", name="same"),
            ],
            control_lags={"same": 1}, unit="unit", time="time",
        )


def test_shared_category_uses_one_nuisance_reference() -> None:
    rng = np.random.default_rng(13)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(20), 10),
        "time": np.tile(np.arange(10), 20),
        "group": np.tile([0, 1], 100),
        **{name: rng.normal(size=200) for name in ("y", "a", "b")},
    })
    options = dict(outcome="y", unit="unit", time="time")
    actual = LocalProjection(horizons=1, covariance="hc0").fit(
        data,
        shock=[
            s.interact("a", by=s.C("group", levels=[0, 1]), name="A"),
            s.interact("b", by=s.C("group", levels=[1, 0]), name="B"),
        ],
        **options,
    )
    expected = LocalProjection(horizons=1, covariance="hc0").fit(
        data,
        shock=[
            s.interact("a", by=s.C("group", levels=[0, 1]), name="A"),
            s.interact("b", by=s.C("group", levels=[0, 1]), name="B"),
        ],
        **options,
    )
    assert "group[0]" not in actual.feature_names_in_
    order = [list(actual.feature_names_in_).index(name) for name in expected.feature_names_in_]
    np.testing.assert_allclose(actual.coef_[:, order], expected.coef_)
    np.testing.assert_allclose(
        actual.covariance_[:, order][:, :, order], expected.covariance_, atol=1e-12
    )


def test_category_without_horizon_support_is_reported() -> None:
    rng = np.random.default_rng(62)
    data = pd.DataFrame({
        "unit": np.repeat(np.arange(20), 5),
        "time": np.tile(np.arange(5), 20),
        "x": rng.normal(size=100),
        "y": rng.normal(size=100),
    })
    data["group"] = (data.time == 4).astype(int)
    with pytest.raises(ValueError, match="horizon 1.*group\\[1\\]"):
        LocalProjection(horizons=1, covariance="hc0", sample="per_horizon").fit(
            data, outcome="y", shock=s.interact("x", by=s.C("group", levels=[0, 1])),
            unit="unit", time="time",
        )
