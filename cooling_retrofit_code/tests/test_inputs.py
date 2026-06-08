from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_utils import (
    _load_heating_demand,
    _series_by_timestamp,
    _with_canonical_slot_timestamp,
    load_config,
    load_input_data,
)
from typical_days import build_time_index


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_config_and_real_input_data_are_loaded_with_expected_columns():
    config = load_config(PROJECT_DIR / "config.json")
    data = load_input_data(config)
    frame = data.hourly

    assert config["typical_days"]["enabled"] is True
    assert config["typical_days"]["k"] == 8
    assert config["typical_days"]["random_seed"] == 42
    assert {"it_load_kw", "outdoor_temp_c", "heating_demand_kw"}.issubset(
        config["typical_days"]["feature_columns"]
    )
    assert config["technology"]["fitted"]["e_rdhx_kw_per_kw"] == 0.0265
    assert config["technology"]["fitted"]["cop_chiller"] == 3.0
    assert config["technology"]["fitted"]["ac_fan_kw_per_kw"] == 0.2137
    assert config["scenario"]["business_growth"]["random_seed"] == 20260601
    assert config["scenario"]["ac_aging"]["random_seed"] == 20260601
    assert len(frame) == 8760
    assert frame["timestamp_hour_utc"].is_unique
    assert frame["timestamp_hour_utc"].is_monotonic_increasing
    assert frame["timestamp_hour_utc"].min() == pd.Timestamp("2025-01-01 00:00:00+00:00")
    assert frame["timestamp_hour_utc"].max() == pd.Timestamp("2025-12-31 23:00:00+00:00")
    assert {
        "it_load_kw",
        "price_yuan_per_kwh",
        "carbon_kg_per_kwh",
        "outdoor_temp_c",
        "heating_demand_kw",
    }.issubset(frame.columns)
    rack_cols = [c for c in frame.columns if "rack_it_" in c and c.endswith("_kw")]
    assert rack_cols, "rack-level IT load columns must be preserved for rack-level dispatch"
    assert np.allclose(
        frame[rack_cols].sum(axis=1).to_numpy(),
        frame["it_load_kw"].to_numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert "it_load_kw_base" in frame.columns
    assert frame["it_load_kw"].max() >= frame["it_load_kw_base"].max()
    assert "psi_amb" in frame.columns
    assert frame["psi_amb"].eq(1.0).all()

    scenario = data.scenario_growth_age
    assert not scenario.empty
    assert {
        "rack_id",
        "zone_id",
        "ac_unit",
        "rho_load",
        "alpha_age",
        "epsilon_age",
        "seed",
        "source",
    }.issubset(scenario.columns)
    assert scenario["seed"].eq(20260601).all()
    assert scenario["source"].eq("simulated").all()
    assert scenario["rho_load"].between(0.05, 0.25).all()
    assert scenario["alpha_age"].between(0.08, 0.25).all()
    assert np.allclose(scenario["epsilon_age"], scenario["alpha_age"])
    old_cap_cols = [c for c in frame.columns if c.startswith("old_ac_capacity_eff_") and c.endswith("_kw")]
    old_coeff_cols = [c for c in frame.columns if c.startswith("old_ac_terminal_coeff_")]
    assert old_cap_cols
    assert old_coeff_cols
    assert frame[old_cap_cols].max().max() < frame[rack_cols].sum(axis=1).max()
    zone_col = "ac_unit" if "ac_unit" in data.rack_metadata.columns else "zone"
    metadata = data.rack_metadata.set_index("rack_id")
    zone_loads = {}
    for col in rack_cols:
        rack_id = col.removeprefix("rack_it_").removesuffix("_kw")
        zone = str(metadata.loc[rack_id, zone_col])
        zone_loads.setdefault(zone, pd.Series(0.0, index=frame.index))
        zone_loads[zone] = zone_loads[zone] + frame[col]
    redundancy = config["scenario"]["existing_cooling_redundancy"]["old_ac_factor"]
    for zone, load in zone_loads.items():
        cap_col = f"old_ac_capacity_eff_{zone}_kw"
        assert cap_col in frame.columns
        assert float(frame[cap_col].min()) + 1.0e-6 >= float(load.max()) * redundancy


def test_input_data_replicates_room_five_times_with_heterogeneous_growth_and_aging():
    config = load_config(PROJECT_DIR / "config.json")
    data = load_input_data(config)
    frame = data.hourly
    scenario = data.scenario_growth_age
    zone_col = "ac_unit" if "ac_unit" in data.rack_metadata.columns else "zone"

    replication_factor = int(config["scenario"]["room_replication_factor"])
    assert replication_factor == 5
    rack_cols = [c for c in frame.columns if "rack_it_" in c and c.endswith("_kw")]
    assert len(rack_cols) == 49 * replication_factor
    assert len(data.rack_metadata) == 49 * replication_factor
    assert "room_id" in data.rack_metadata.columns
    assert data.rack_metadata["room_id"].nunique() == replication_factor
    assert data.rack_metadata[zone_col].nunique() == 6 * replication_factor

    assert len(scenario) == 49 * replication_factor
    assert "room_id" in scenario.columns
    assert scenario["room_id"].nunique() == replication_factor
    assert scenario["zone_id"].nunique() == 6 * replication_factor
    assert scenario["rho_load"].nunique() > 49
    assert scenario.groupby("zone_id")["alpha_age"].nunique().max() == 1
    assert scenario["alpha_age"].nunique() > 6
    assert scenario.groupby("room_id")["alpha_age"].nunique().min() >= 2


def test_timestamp_and_numeric_helpers_reject_bad_inputs():
    duplicate_frame = pd.DataFrame(
        {
            "timestamp_hour_utc": pd.to_datetime(
                ["2025-01-01 00:00:00+00:00", "2025-01-01 00:00:00+00:00"],
                utc=True,
            ),
            "value": [1.0, 3.0],
        }
    )
    bad_numeric_frame = duplicate_frame.assign(
        timestamp_hour_utc=pd.to_datetime(
            ["2025-01-01 00:00:00+00:00", "2025-01-01 01:00:00+00:00"],
            utc=True,
        ),
        value=["1.0", "bad"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        _series_by_timestamp(
            duplicate_frame,
            "value",
            "value_out",
            allow_duplicate_timestamp_mean=False,
            source_name="test_source",
        )
    averaged = _series_by_timestamp(
        duplicate_frame,
        "value",
        "value_out",
        allow_duplicate_timestamp_mean=True,
        source_name="test_source",
    )
    assert averaged["value_out"].tolist() == [2.0]

    with pytest.raises(ValueError, match="Non-numeric"):
        _series_by_timestamp(
            bad_numeric_frame,
            "value",
            "value_out",
            allow_duplicate_timestamp_mean=False,
            source_name="test_source",
        )
    with pytest.raises(ValueError, match="hourly"):
        _with_canonical_slot_timestamp(
            pd.DataFrame({"timestamp": ["2025-01-01T00:30:00+08:00"]}),
            ("timestamp",),
        )


def test_heating_demand_loader_prefers_chinese_heat_column_and_rejects_ambiguous_schema():
    valid = pd.DataFrame(
        {
            "Date/Time": ["Year 1 Jan 01 1:00"],
            "\u51b7\u8d1f\u8377/kWh": [11.0],
            "\u70ed\u8d1f\u8377/kWh": [22.0],
            "\u7535\u8d1f\u8377/kWh": [33.0],
            "timestamp_hour_utc": pd.to_datetime(["2025-01-01 00:00:00+00:00"], utc=True),
        }
    )
    invalid = pd.DataFrame(
        {
            "Date/Time": ["Year 1 Jan 01 1:00"],
            "load_a": [1.0],
            "load_b": [2.0],
            "load_c": [3.0],
            "timestamp_hour_utc": pd.to_datetime(["2025-01-01 00:00:00+00:00"], utc=True),
        }
    )

    assert _load_heating_demand(valid)["heating_demand_kw"].tolist() == [22.0]
    with pytest.raises(ValueError, match="Heating demand"):
        _load_heating_demand(invalid)


def test_heating_demand_is_loaded_without_it_peak_scaling():
    config = load_config(PROJECT_DIR / "config.json")
    data = load_input_data(config)
    frame = data.hourly

    legacy_config_key = "heat" + "_demand" + "_scaling"
    legacy_columns = {"heating_demand_" + "raw_kw", "heating_demand_" + "scale_factor"}
    assert legacy_config_key not in config.get("scenario", {})
    assert legacy_columns.isdisjoint(frame.columns)

    raw = _load_heating_demand(data.heating_demand)
    merged = frame[["timestamp_hour_utc", "heating_demand_kw"]].merge(
        raw,
        on="timestamp_hour_utc",
        suffixes=("", "_raw"),
    )
    np.testing.assert_allclose(
        merged["heating_demand_kw"].to_numpy(dtype=float),
        merged["heating_demand_kw_raw"].to_numpy(dtype=float),
    )


def test_kmeans_typical_days_selects_eight_centroids_and_optional_peak_day_from_real_data():
    config = load_config(PROJECT_DIR / "config.json")
    data = load_input_data(config)

    result = build_time_index(data.hourly, config["typical_days"])

    assert result.mode == "typical_days"
    assert 8 <= len(result.representative_days) <= 9
    assert result.frame["representative_day_id"].nunique() == len(result.representative_days)
    assert result.day_weights.sum() == data.hourly["timestamp_hour_utc"].dt.floor("D").nunique()
    assert set(result.frame["timestamp_hour_utc"]).issubset(set(data.hourly["timestamp_hour_utc"]))


def test_typical_days_preserve_weights_and_can_fall_back_to_full_timeseries():
    rows = []
    for day_offset, value in enumerate([0.0, 10.0, 11.0, 12.0]):
        for hour in range(24):
            rows.append(
                {
                    "timestamp_hour_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=day_offset, hours=hour),
                    "load_kw": value,
                }
            )
    hourly = pd.DataFrame(rows)

    typical = build_time_index(
        hourly,
        {"enabled": True, "k": 2, "feature_columns": ["load_kw"], "random_seed": 0},
    )
    full = build_time_index(hourly, {"enabled": False})

    assert typical.representative_days == sorted(typical.representative_days)
    assert sorted(typical.day_weights.tolist()) == [1, 3]
    assert typical.day_weights.sum() == 4
    assert full.mode == "full_timeseries"
    assert len(full.frame) == len(hourly)


def test_typical_days_append_peak_day_when_centroid_selection_misses_it():
    rows = []
    for day_offset, value in enumerate([0.0, 1.0, 2.0, 100.0]):
        for hour in range(24):
            rows.append(
                {
                    "timestamp_hour_utc": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=day_offset, hours=hour),
                    "it_load_kw": value,
                }
            )
    hourly = pd.DataFrame(rows)

    result = build_time_index(
        hourly,
        {
            "enabled": True,
            "k": 1,
            "feature_columns": ["it_load_kw"],
            "random_seed": 0,
            "append_peak_day": True,
            "peak_column": "it_load_kw",
        },
    )

    peak_day = pd.Timestamp("2025-01-04", tz="UTC")
    assert peak_day in result.representative_days
    assert len(result.representative_days) == 2
    assert result.day_weights.loc[peak_day] == 1
    assert result.day_weights.sum() == 4


def test_typical_day_builder_rejects_missing_features_and_malformed_days():
    hourly = pd.DataFrame(
        {
            "timestamp_hour_utc": pd.date_range("2025-01-01", periods=24, freq="h", tz="UTC"),
            "available_feature": range(24),
        }
    )
    with pytest.raises(ValueError, match="missing_feature"):
        build_time_index(
            hourly,
            {
                "enabled": True,
                "k": 1,
                "feature_columns": ["available_feature", "missing_feature"],
                "random_seed": 0,
            },
        )

    malformed = pd.DataFrame(
        [
            {"timestamp_hour_utc": pd.Timestamp("2025-03-01", tz="UTC") + pd.Timedelta(hours=h), "load_kw": 100.0}
            for h in [0, 0, *range(1, 23)]
        ]
        + [
            {"timestamp_hour_utc": pd.Timestamp("2025-03-02", tz="UTC") + pd.Timedelta(hours=h), "load_kw": 1.0}
            for h in range(24)
        ]
    )
    result = build_time_index(
        malformed,
        {"enabled": True, "k": 1, "feature_columns": ["load_kw"], "random_seed": 0},
    )
    assert result.representative_days == [pd.Timestamp("2025-03-02", tz="UTC")]
