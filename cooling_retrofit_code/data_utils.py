from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CONFIG_SECTIONS = {
    "paths",
    "scenario",
    "typical_days",
    "solver",
    "technology",
    "economics",
    "carbon",
    "weight",
    "assumptions",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    missing = REQUIRED_CONFIG_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    return config


@dataclass(frozen=True)
class InputData:
    rack_metadata: pd.DataFrame
    heat_matrix: pd.DataFrame
    time_series: pd.DataFrame
    price: pd.DataFrame
    carbon: pd.DataFrame
    weather: pd.DataFrame
    heating_demand: pd.DataFrame
    ac_hourly: pd.DataFrame
    hourly: pd.DataFrame
    scenario_growth_age: pd.DataFrame


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def _timestamp_col(df: pd.DataFrame) -> str:
    for col in ("timestamp_hour_utc", "timestamp_utc", "timestamp", "time", "datetime", "hour"):
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found in columns: {list(df.columns)}")


def _numeric_col(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    fallback_contains: tuple[str, ...],
) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    lower = {col: col.lower() for col in df.columns}
    for col, name in lower.items():
        if all(token in name for token in fallback_contains):
            return col
    raise ValueError(f"No numeric column found in columns: {list(df.columns)}")


def _with_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = _timestamp_col(out)
    out["timestamp_hour_utc"] = pd.to_datetime(out[col], utc=True).dt.floor("h")
    return out


def _with_canonical_slot_timestamp(
    df: pd.DataFrame,
    preferred_columns: tuple[str, ...],
    *,
    source_name: str = "source",
    require_preferred_column: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    source_col = next((col for col in preferred_columns if col in out.columns), None)
    if source_col is None:
        if require_preferred_column:
            required = " or ".join(preferred_columns)
            raise ValueError(f"{source_name} requires timestamp column: {required}")
        source_col = _timestamp_col(out)
    timestamp_text = out[source_col].astype(str).str.strip()
    parts = timestamp_text.str.extract(
        r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})[T ]"
        r"(?P<hour>\d{2}):(?P<minute>\d{2})"
        r"(?::(?P<second>\d{2})(?:\.\d+)?)?"
        r"(?P<tz>Z|[+-]\d{2}:\d{2})?$"
    )
    if parts.isna().any(axis=None):
        raise ValueError(f"Could not parse full timestamp slots from column: {source_col}")
    if not ((parts["minute"] == "00") & (parts["second"].fillna("00") == "00")).all():
        raise ValueError(f"Timestamps in {source_col} must be hourly")
    parts = parts[["month", "day", "hour"]].astype(int)
    keep = ~((parts["month"] == 2) & (parts["day"] == 29))
    out = out.loc[keep].copy()
    parts = parts.loc[keep]
    slot_text = (
        "2025-"
        + parts["month"].astype(str).str.zfill(2)
        + "-"
        + parts["day"].astype(str).str.zfill(2)
        + " "
        + parts["hour"].astype(str).str.zfill(2)
        + ":00:00+00:00"
    )
    out["timestamp_hour_utc"] = pd.to_datetime(slot_text, utc=True)
    return out


def _with_carbon_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hour_of_year" not in out.columns:
        raise ValueError("Carbon data must include hour_of_year")
    hour_of_year = pd.to_numeric(out["hour_of_year"], errors="raise")
    if len(out) != 8760:
        raise ValueError(f"Carbon data must have exactly 8760 rows, found {len(out)}")
    if not hour_of_year.is_unique:
        raise ValueError("Carbon hour_of_year values must be unique")
    if hour_of_year.min() != 1 or hour_of_year.max() != 8760:
        raise ValueError("Carbon hour_of_year must span 1..8760")
    out["timestamp_hour_utc"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        hour_of_year - 1, unit="h"
    )
    return out


def _expected_heating_datetime_labels() -> pd.Series:
    end_times = pd.date_range("2025-01-01 01:00", periods=8760, freq="h")
    labels = []
    for end_time in end_times:
        year_label = "Year 2" if end_time.year == 2026 else "Year 1"
        labels.append(f"{year_label} {end_time:%b} {end_time:%d} {end_time.hour}:00")
    return pd.Series(labels)


def _with_heating_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "Date/Time" not in df.columns:
        raise ValueError("Heating demand Date/Time column is required")
    if len(df) != 8760:
        raise ValueError(f"Heating demand data must have exactly 8760 rows, found {len(df)}")
    out = df.copy()
    labels = out["Date/Time"]
    if labels.isna().any():
        raise ValueError("Heating demand Date/Time labels must be present")
    label_text = labels.astype(str).str.strip()
    if (label_text == "").any():
        raise ValueError("Heating demand Date/Time labels must be present")
    if not label_text.is_unique:
        raise ValueError("Heating demand Date/Time labels must be unique")
    expected_labels = _expected_heating_datetime_labels()
    mismatches = label_text.reset_index(drop=True) != expected_labels
    if mismatches.any():
        row = int(mismatches.idxmax())
        raise ValueError(
            f"Heating demand Date/Time label mismatch at row {row}: "
            f"expected {expected_labels.iloc[row]!r}, found {label_text.iloc[row]!r}"
        )
    # EnergyPlus-style labels are end-of-hour; row order supplies dispatch intervals.
    # Row 0 is aligned to canonical Jan 1 00:00.
    out["timestamp_hour_utc"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(
        out.index, unit="h"
    )
    return out


def _validated_numeric_series(
    df: pd.DataFrame,
    value_col: str,
    source_name: str,
) -> pd.Series:
    values = pd.to_numeric(df[value_col], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values.to_numpy(dtype=float))
    if invalid.any():
        count = int(invalid.sum())
        raise ValueError(
            f"Non-numeric, null, or non-finite values in {source_name}.{value_col}: {count}"
        )
    return values


def _series_by_timestamp(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    *,
    allow_duplicate_timestamp_mean: bool,
    source_name: str,
) -> pd.DataFrame:
    out = df.copy()
    if out["timestamp_hour_utc"].isna().any():
        raise ValueError(f"Null timestamp_hour_utc values in {source_name}")
    out[output_col] = _validated_numeric_series(out, value_col, source_name)
    if out["timestamp_hour_utc"].duplicated().any():
        if not allow_duplicate_timestamp_mean:
            raise ValueError(f"Found duplicate timestamp_hour_utc values in {source_name}")
        # IT and weather files intentionally contain multi-year samples per canonical slot.
        return out.groupby("timestamp_hour_utc", as_index=False)[output_col].mean()
    return out[["timestamp_hour_utc", output_col]]


def _rack_loads_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    rack_cols = _rack_load_columns(df)
    if not rack_cols:
        return df[["timestamp_hour_utc"]].drop_duplicates().copy()

    out = df[["timestamp_hour_utc", *rack_cols]].copy()
    for col in rack_cols:
        out[col] = _validated_numeric_series(out, col, "time_series")
    if out["timestamp_hour_utc"].duplicated().any():
        return out.groupby("timestamp_hour_utc", as_index=False)[rack_cols].mean()
    return out


def _rack_load_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if "rack_it_" in col and col.endswith("_kw")]


def _rack_id_from_load_col(col: str) -> str:
    return col.removesuffix("_kw")


def _zone_column(rack_metadata: pd.DataFrame) -> str | None:
    for col in ("ac_unit", "zone"):
        if col in rack_metadata.columns:
            return col
    return None


def _scenario_seed(config: dict[str, Any]) -> int:
    growth = config.get("scenario", {}).get("business_growth", {})
    aging = config.get("scenario", {}).get("ac_aging", {})
    return int(growth.get("random_seed", aging.get("random_seed", 20260601)))


def _room_replication_factor(config: dict[str, Any]) -> int:
    value = config.get("scenario", {}).get("room_replication_factor", 1)
    factor = int(round(float(value)))
    if factor < 1:
        raise ValueError("scenario.room_replication_factor must be >= 1")
    return factor


def _replicated_name(room_id: str, value: Any) -> str:
    return f"{room_id}_{value}"


def _replicate_heat_matrix(
    heat_matrix: pd.DataFrame,
    original_rack_count: int,
    factor: int,
    rack_ids: list[str],
) -> pd.DataFrame:
    if factor <= 1:
        return heat_matrix
    values = pd.to_numeric(heat_matrix.stack(), errors="coerce").unstack().to_numpy(dtype=float)
    if values.shape[0] != original_rack_count or values.shape[1] != original_rack_count:
        return heat_matrix
    block = np.kron(np.eye(factor), np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))
    return pd.DataFrame(block, index=rack_ids, columns=rack_ids)


def _replicate_room_inputs(
    hourly: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    heat_matrix: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor = _room_replication_factor(config)
    if factor <= 1:
        return hourly, rack_metadata, heat_matrix

    rack_cols = _rack_load_columns(hourly)
    if not rack_cols:
        return hourly, rack_metadata, heat_matrix

    base_hourly = hourly.drop(columns=rack_cols).copy()
    replicated_columns: dict[str, np.ndarray] = {}
    metadata_frames: list[pd.DataFrame] = []
    replicated_rack_ids: list[str] = []
    zone_col = _zone_column(rack_metadata)

    for room_idx in range(1, factor + 1):
        room_id = f"room{room_idx:02d}"
        for col in rack_cols:
            replicated_columns[f"{room_id}_{col}"] = hourly[col].to_numpy()

        metadata = rack_metadata.copy()
        metadata["room_id"] = room_id
        metadata["room_index"] = room_idx
        if "rack_id" in metadata.columns:
            metadata["base_rack_id"] = metadata["rack_id"].astype(str)
            metadata["rack_id"] = metadata["rack_id"].map(lambda value: _replicated_name(room_id, value))
        if zone_col:
            metadata[f"base_{zone_col}"] = metadata[zone_col].astype(str)
            metadata[zone_col] = metadata[zone_col].map(lambda value: _replicated_name(room_id, value))
        other_zone_col = "zone" if zone_col == "ac_unit" else "ac_unit"
        if other_zone_col in metadata.columns:
            metadata[f"base_{other_zone_col}"] = metadata[other_zone_col].astype(str)
            metadata[other_zone_col] = metadata[other_zone_col].map(lambda value: _replicated_name(room_id, value))
        metadata_frames.append(metadata)
        replicated_rack_ids.extend(metadata["rack_id"].astype(str).tolist())

    replicated_hourly = pd.concat(
        [base_hourly, pd.DataFrame(replicated_columns, index=hourly.index)],
        axis=1,
    )
    replicated_rack_cols = _rack_load_columns(replicated_hourly)
    replicated_hourly["it_load_kw"] = replicated_hourly[replicated_rack_cols].sum(axis=1)
    replicated_metadata = pd.concat(metadata_frames, ignore_index=True)
    replicated_heat_matrix = _replicate_heat_matrix(
        heat_matrix,
        original_rack_count=len(rack_cols),
        factor=factor,
        rack_ids=replicated_rack_ids,
    )
    return replicated_hourly, replicated_metadata, replicated_heat_matrix


def _build_scenario_growth_age_table(
    rack_metadata: pd.DataFrame,
    rack_cols: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    seed = _scenario_seed(config)
    rng = np.random.default_rng(seed)
    growth = config.get("scenario", {}).get("business_growth", {})
    aging = config.get("scenario", {}).get("ac_aging", {})
    rho_min = float(growth.get("rho_load_min", 0.05))
    rho_max = float(growth.get("rho_load_max", 0.25))
    alpha_min = float(aging.get("alpha_age_min", 0.08))
    alpha_max = float(aging.get("alpha_age_max", 0.25))
    if rho_min > rho_max:
        raise ValueError("rho_load_min must be <= rho_load_max")
    if alpha_min > alpha_max:
        raise ValueError("alpha_age_min must be <= alpha_age_max")

    zone_col = _zone_column(rack_metadata)
    metadata = rack_metadata.set_index("rack_id") if "rack_id" in rack_metadata.columns else pd.DataFrame()
    rack_ids = [_rack_id_from_load_col(col) for col in rack_cols]
    zone_by_rack: dict[str, str] = {}
    room_by_rack: dict[str, str] = {}
    for rack_id in rack_ids:
        zone = "zone_1"
        room_id = "room01"
        if zone_col and rack_id in metadata.index:
            zone = str(metadata.loc[rack_id, zone_col])
        if "room_id" in metadata.columns and rack_id in metadata.index:
            room_id = str(metadata.loc[rack_id, "room_id"])
        zone_by_rack[rack_id] = zone
        room_by_rack[rack_id] = room_id

    alpha_by_zone = {
        zone: float(rng.uniform(alpha_min, alpha_max))
        for zone in sorted(set(zone_by_rack.values()))
    }
    rows: list[dict[str, Any]] = []
    for rack_id, rack_col in zip(rack_ids, rack_cols):
        zone = zone_by_rack[rack_id]
        rho = float(rng.uniform(rho_min, rho_max))
        alpha = alpha_by_zone[zone]
        rows.append(
            {
                "rack_id": rack_id,
                "rack_col": rack_col,
                "zone_id": zone,
                "ac_unit": zone,
                "room_id": room_by_rack[rack_id],
                "rho_load": rho,
                "alpha_age": alpha,
                "epsilon_age": alpha,
                "seed": seed,
                "source": "simulated",
            }
        )
    return pd.DataFrame(rows)


def _elapsed_years_for_growth(hourly: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    growth = config.get("scenario", {}).get("business_growth", {})
    horizon = float(growth.get("planning_horizon_years", growth.get("horizon_years", 1.0)))
    timestamps = pd.to_datetime(hourly["timestamp_hour_utc"], utc=True)
    elapsed = (timestamps - timestamps.min()).dt.total_seconds().to_numpy(dtype=float)
    span = float(np.max(elapsed)) if len(elapsed) else 0.0
    if span <= 0.0:
        return np.zeros(len(hourly), dtype=float)
    return horizon * elapsed / span


def _rack_it_capacity_kw(rack_metadata: pd.DataFrame, rack_id: str, fallback: float) -> float:
    if "rack_id" not in rack_metadata.columns:
        return fallback
    metadata = rack_metadata.set_index("rack_id")
    if rack_id not in metadata.index:
        return fallback
    row = metadata.loc[rack_id]
    for col in ("q_it_max_kw", "it_capacity_kw", "rated_power_kw", "rack_it_max_kw"):
        if col in metadata.columns:
            value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0.0:
                return float(value)
    return fallback


def _zone_peak_from_rack_columns(
    hourly: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    rack_cols: list[str],
) -> dict[str, float]:
    zone_col = _zone_column(rack_metadata)
    metadata = rack_metadata.set_index("rack_id") if "rack_id" in rack_metadata.columns else pd.DataFrame()
    zone_series: dict[str, pd.Series] = {}
    for col in rack_cols:
        rack_id = _rack_id_from_load_col(col)
        zone = "zone_1"
        if zone_col and rack_id in metadata.index:
            zone = str(metadata.loc[rack_id, zone_col])
        zone_series.setdefault(zone, pd.Series(0.0, index=hourly.index))
        zone_series[zone] = zone_series[zone] + pd.to_numeric(hourly[col], errors="coerce").fillna(0.0)
    return {zone: float(series.max()) for zone, series in zone_series.items()}


def _old_ac_capacity_config(config: dict[str, Any]) -> dict[str, float]:
    tech = config.get("technology", {})
    for key in (
        "old_ac_capacity_kw_by_ac_unit",
        "old_ac_capacity_by_ac_unit_kw",
        "old_ac_capacity_by_zone_kw",
        "existing_ac_capacity_kw_by_zone",
    ):
        value = tech.get(key)
        if isinstance(value, dict):
            return {str(zone): float(capacity) for zone, capacity in value.items()}
    return {}


def _apply_business_growth_and_ac_aging(
    hourly: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    scenario_table: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = hourly.copy()
    rack_cols = _rack_load_columns(out)
    extra_columns: dict[str, Any] = {"psi_amb": np.ones(len(out), dtype=float)}
    if not rack_cols:
        return pd.concat([out, pd.DataFrame(extra_columns, index=out.index)], axis=1).copy()

    baseline_zone_peak = _zone_peak_from_rack_columns(out, rack_metadata, rack_cols)
    years = _elapsed_years_for_growth(out, config)
    extra_columns["it_load_kw_base"] = pd.to_numeric(out["it_load_kw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scenario_by_rack = scenario_table.set_index("rack_id")
    for col in rack_cols:
        rack_id = _rack_id_from_load_col(col)
        base = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        rho = float(scenario_by_rack.loc[rack_id, "rho_load"]) if rack_id in scenario_by_rack.index else 0.0
        fallback_capacity = float(base.max() * ((1.0 + rho) ** max(float(years.max()), 0.0)))
        qmax = _rack_it_capacity_kw(rack_metadata, rack_id, fallback_capacity)
        extra_columns[f"{col}_base"] = base
        out[col] = np.minimum(base * np.power(1.0 + rho, years), qmax)

    out["it_load_kw"] = out[rack_cols].sum(axis=1)
    old_ac_config = _old_ac_capacity_config(config)
    tech = config.get("technology", {})
    fitted = tech.get("fitted", {})
    e_air_base = float(fitted.get("ac_fan_kw_per_kw", tech.get("ac_fan_kw_per_kw", 0.2137)))
    for zone, group in scenario_table.groupby("zone_id", sort=True):
        alpha = float(group["alpha_age"].iloc[0])
        epsilon = float(group["epsilon_age"].iloc[0])
        old_capacity = old_ac_config.get(str(zone), baseline_zone_peak.get(str(zone), 0.0))
        effective_capacity = max(0.0, (1.0 - alpha) * float(old_capacity))
        effective_coeff = e_air_base / max(1.0 - epsilon, 1.0e-6)
        extra_columns[f"old_ac_capacity_eff_{zone}_kw"] = np.full(len(out), effective_capacity)
        extra_columns[f"old_ac_terminal_coeff_{zone}_kw_per_kw"] = np.full(len(out), effective_coeff)
        extra_columns[f"old_ac_alpha_age_{zone}"] = np.full(len(out), alpha)
        extra_columns[f"old_ac_epsilon_age_{zone}"] = np.full(len(out), epsilon)
        extra_columns[f"psi_amb_{zone}"] = np.ones(len(out), dtype=float)
    return pd.concat([out, pd.DataFrame(extra_columns, index=out.index)], axis=1).copy()


def _attach_scenario_to_rack_metadata(
    rack_metadata: pd.DataFrame,
    scenario_table: pd.DataFrame,
) -> pd.DataFrame:
    if scenario_table.empty or "rack_id" not in rack_metadata.columns:
        return rack_metadata
    columns = ["rack_id", "rho_load", "alpha_age", "epsilon_age", "seed", "source"]
    scenario = scenario_table[columns].copy()
    merged = rack_metadata.merge(scenario, on="rack_id", how="left", suffixes=("", "_scenario"))
    for col in ("rho_load", "alpha_age", "epsilon_age"):
        scenario_col = f"{col}_scenario"
        if scenario_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[scenario_col])
            merged = merged.drop(columns=[scenario_col])
    return merged


def _hour_index_series(
    df: pd.DataFrame,
    value_col: str,
    output_col: str,
    *,
    source_name: str,
) -> pd.DataFrame:
    out = df.copy()
    if out["timestamp_hour_utc"].isna().any():
        raise ValueError(f"Null timestamp_hour_utc values in {source_name}")
    if out["timestamp_hour_utc"].duplicated().any():
        raise ValueError(f"Found duplicate timestamp_hour_utc values in {source_name}")
    out[output_col] = _validated_numeric_series(out, value_col, source_name)
    return out[["timestamp_hour_utc", output_col]]


def _load_heating_demand(df: pd.DataFrame) -> pd.DataFrame:
    candidates = (
        "heating_demand_kw",
        "heating_demand_kwh",
        "heat_load_kw",
        "\u70ed\u8d1f\u8377/kWh",
        "\u70ed\u8d1f\u8377",
    )
    heating_col = next((col for col in candidates if col in df.columns), None)
    non_timestamp_columns = [col for col in df.columns if col != "timestamp_hour_utc"]
    has_energyplus_schema = len(non_timestamp_columns) == 4 and non_timestamp_columns[0] == "Date/Time"
    if has_energyplus_schema:
        load_columns = non_timestamp_columns[1:]
        for col in load_columns:
            _validated_numeric_series(df, col, "heating_demand")
    elif heating_col is None:
        raise ValueError(f"Heating demand schema is not recognized: {list(df.columns)}")
    if heating_col is None:
        raise ValueError(
            "Heating demand schema is recognized, but no known heating load column was found"
        )
    return _hour_index_series(
        df,
        heating_col,
        "heating_demand_kw",
        source_name="heating_demand",
    )


def _scale_heating_demand_to_it_load(hourly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    scaling = config.get("scenario", {}).get("heat_demand_scaling", {})
    if not scaling.get("enabled", False):
        return hourly

    method = scaling.get("method", "peak_ratio_to_it_load")
    if method != "peak_ratio_to_it_load":
        raise ValueError(f"Unsupported heat demand scaling method: {method}")

    target_fraction = float(scaling.get("target_peak_fraction_of_it_load", 1.0))
    if target_fraction < 0:
        raise ValueError("target_peak_fraction_of_it_load must be non-negative")

    it_peak = float(hourly["it_load_kw"].max())
    heat_peak = float(hourly["heating_demand_kw"].max())
    out = hourly.copy()
    out["heating_demand_raw_kw"] = out["heating_demand_kw"]

    if it_peak <= 0 or heat_peak <= 0:
        out["heating_demand_scale_factor"] = 0.0
        out["heating_demand_kw"] = 0.0
        return out

    scale_factor = target_fraction * it_peak / heat_peak
    if not scaling.get("allow_upscale", False):
        scale_factor = min(scale_factor, 1.0)

    out["heating_demand_scale_factor"] = scale_factor
    out["heating_demand_kw"] = out["heating_demand_kw"] * scale_factor
    return out


def _validate_hourly(hourly: pd.DataFrame) -> None:
    required = (
        "timestamp_hour_utc",
        "it_load_kw",
        "price_yuan_per_kwh",
        "carbon_kg_per_kwh",
        "outdoor_temp_c",
        "heating_demand_kw",
    )
    missing = [col for col in required if col not in hourly.columns]
    if missing:
        raise ValueError(f"Aligned hourly data missing required columns: {missing}")
    if len(hourly) != 8760:
        raise ValueError(f"Aligned hourly data must have exactly 8760 rows, found {len(hourly)}")
    if not hourly["timestamp_hour_utc"].is_unique:
        raise ValueError("Aligned hourly timestamps must be unique")
    if not hourly["timestamp_hour_utc"].is_monotonic_increasing:
        raise ValueError("Aligned hourly timestamps must be sorted")
    expected_min = pd.Timestamp("2025-01-01 00:00:00+00:00")
    expected_max = pd.Timestamp("2025-12-31 23:00:00+00:00")
    if hourly["timestamp_hour_utc"].min() != expected_min:
        raise ValueError("Aligned hourly data must start at 2025-01-01 00:00:00+00:00")
    if hourly["timestamp_hour_utc"].max() != expected_max:
        raise ValueError("Aligned hourly data must end at 2025-12-31 23:00:00+00:00")
    null_columns = [col for col in required if hourly[col].isna().any()]
    if null_columns:
        raise ValueError(f"Aligned hourly data has nulls in columns: {null_columns}")


def load_input_data(config: dict[str, Any]) -> InputData:
    data_dir = Path(config["paths"]["data_dir"])

    rack_metadata = _read_csv(data_dir, "rack_metadata.csv")
    heat_matrix = pd.read_csv(data_dir / "H_matrix.csv", header=None)
    time_series = _with_canonical_slot_timestamp(
        _read_csv(data_dir, "time_series.csv"),
        ("hour", "timestamp_hour_utc", "timestamp_utc", "timestamp", "time", "datetime"),
    )
    price = _with_canonical_slot_timestamp(
        _read_csv(
            data_dir,
            "price_grid_beijing_2025_two_part_35_110kv_hourly_yuan_per_kwh.csv",
        ),
        ("timestamp_local",),
        source_name="price",
        require_preferred_column=True,
    )
    carbon = _with_carbon_hour_timestamp(
        _read_csv(data_dir, "carbon_factor_China_2025_kgCO2_per_kWh.csv")
    )
    weather = _with_canonical_slot_timestamp(
        _read_csv(data_dir, "weather_open_meteo_hourly.csv"),
        ("timestamp_utc", "timestamp_hour_utc", "timestamp", "time", "datetime"),
    )
    heating_demand = _with_heating_hour_timestamp(_read_csv(data_dir, "heating_demand.csv"))
    ac_hourly = _with_hour_timestamp(_read_csv(data_dir, "m100_22-07_ac_unit_hourly.csv"))

    load_col = _numeric_col(
        time_series,
        ("it_load_kw", "it_total_kw", "total_it_load_kw", "load_kw"),
        ("it", "kw"),
    )
    price_col = _numeric_col(
        price,
        ("price_yuan_per_kwh", "price_grid_yuan_per_kwh", "tariff_yuan_per_kwh"),
        ("yuan", "kwh"),
    )
    carbon_col = _numeric_col(
        carbon,
        ("carbon_kg_per_kwh", "Beijing_kgCO2_per_kWh", "Mainland_China_kgCO2_per_kWh"),
        ("kgco2", "kwh"),
    )
    weather_col = _numeric_col(
        weather,
        ("outdoor_temp_c", "temperature_2m_c", "dry_bulb_temp_c", "temp_c"),
        ("temp", "c"),
    )

    hourly_parts = [
        _series_by_timestamp(
            time_series,
            load_col,
            "it_load_kw",
            allow_duplicate_timestamp_mean=True,
            source_name="time_series",
        ),
        _rack_loads_by_timestamp(time_series),
        _series_by_timestamp(
            price,
            price_col,
            "price_yuan_per_kwh",
            allow_duplicate_timestamp_mean=False,
            source_name="price",
        ),
        _hour_index_series(
            carbon,
            carbon_col,
            "carbon_kg_per_kwh",
            source_name="carbon",
        ),
        _series_by_timestamp(
            weather,
            weather_col,
            "outdoor_temp_c",
            allow_duplicate_timestamp_mean=True,
            source_name="weather",
        ),
        _load_heating_demand(heating_demand),
    ]
    hourly = hourly_parts[0]
    for part in hourly_parts[1:]:
        hourly = hourly.merge(part, on="timestamp_hour_utc", how="inner")
    hourly = hourly.sort_values("timestamp_hour_utc").reset_index(drop=True)
    hourly, rack_metadata, heat_matrix = _replicate_room_inputs(
        hourly,
        rack_metadata,
        heat_matrix,
        config,
    )
    rack_cols = _rack_load_columns(hourly)
    scenario_growth_age = _build_scenario_growth_age_table(rack_metadata, rack_cols, config)
    rack_metadata = _attach_scenario_to_rack_metadata(rack_metadata, scenario_growth_age)
    hourly = _apply_business_growth_and_ac_aging(hourly, rack_metadata, scenario_growth_age, config)
    hourly = _scale_heating_demand_to_it_load(hourly, config)
    _validate_hourly(hourly)

    return InputData(
        rack_metadata=rack_metadata,
        heat_matrix=heat_matrix,
        time_series=time_series,
        price=price,
        carbon=carbon,
        weather=weather,
        heating_demand=heating_demand,
        ac_hourly=ac_hourly,
        hourly=hourly,
        scenario_growth_age=scenario_growth_age,
    )
