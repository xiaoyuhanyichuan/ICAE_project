# Cooling Retrofit Code Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable codebase for the cooling retrofit planning model in `D:\paper\cooling_retrofit_code\`.

**Architecture:** Create a clean Python project that keeps the old project's two-level algorithm shape but rewrites the domain model around seven zone retrofit configurations, k-means representative days, repair/screening, and a Gurobi inner MILP. Data, decisions, repair, dispatch, evaluation, optimization, and reporting stay in separate focused modules.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, optional gurobipy for MILP execution.

---

## Scope Notes

- Source specification: `D:\paper\docs\superpowers\specs\2026-05-28-cooling-retrofit-code-design.md`.
- Target project: `D:\paper\cooling_retrofit_code\`.
- Real data directory: `D:\paper\real_data\`.
- Reference-only old code: `D:\paper\原研究问题\原来的代码\`.
- `D:\paper` is currently not a git repository. Replace commit checkpoints with test checkpoints unless the user initializes git.

## File Structure

Create these files:

- `D:\paper\cooling_retrofit_code\README.md`: run instructions and data requirements.
- `D:\paper\cooling_retrofit_code\config.json`: default paths, solver, scenario, technology, economics, carbon, weight, and assumptions.
- `D:\paper\cooling_retrofit_code\main.py`: command-line entry point.
- `D:\paper\cooling_retrofit_code\data_utils.py`: config loading, CSV loading, alignment, source tags.
- `D:\paper\cooling_retrofit_code\typical_days.py`: k-means centroid representative-day selection.
- `D:\paper\cooling_retrofit_code\decision.py`: planning decision encoding, bounds, decode, derived flags.
- `D:\paper\cooling_retrofit_code\repair.py`: individual repair and cheap screening.
- `D:\paper\cooling_retrofit_code\feasibility_oracle.py`: screening result wrapper and violation aggregation.
- `D:\paper\cooling_retrofit_code\inner_dispatch.py`: Gurobi MILP builder and dispatch result.
- `D:\paper\cooling_retrofit_code\model.py`: evaluation pipeline and TLCC/TCE aggregation.
- `D:\paper\cooling_retrofit_code\nsga2.py`: NSGA-II with Deb constraint comparison.
- `D:\paper\cooling_retrofit_code\run_optimization.py`: orchestration and result persistence.
- `D:\paper\cooling_retrofit_code\analyze_results.py`: CSV/JSON summary and Chinese Markdown report.
- `D:\paper\cooling_retrofit_code\plotting.py`: figures.
- `D:\paper\cooling_retrofit_code\docs\model_mapping.md`: paper symbol to code mapping.
- `D:\paper\cooling_retrofit_code\docs\assumptions.md`: parameter sources.
- `D:\paper\cooling_retrofit_code\tests\test_data_utils.py`
- `D:\paper\cooling_retrofit_code\tests\test_typical_days.py`
- `D:\paper\cooling_retrofit_code\tests\test_decision_schema.py`
- `D:\paper\cooling_retrofit_code\tests\test_repair.py`
- `D:\paper\cooling_retrofit_code\tests\test_inner_dispatch_small.py`
- `D:\paper\cooling_retrofit_code\tests\test_evaluate_solution.py`

## Task 1: Project Scaffold And Default Config

**Files:**
- Create: `D:\paper\cooling_retrofit_code\README.md`
- Create: `D:\paper\cooling_retrofit_code\config.json`
- Create: `D:\paper\cooling_retrofit_code\tests\test_data_utils.py`
- Create: `D:\paper\cooling_retrofit_code\data_utils.py`

- [ ] **Step 1: Create project directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path "D:\paper\cooling_retrofit_code\tests","D:\paper\cooling_retrofit_code\docs","D:\paper\cooling_retrofit_code\results" | Out-Null
```

Expected: directories exist.

- [ ] **Step 2: Write the failing config test**

Add this test to `D:\paper\cooling_retrofit_code\tests\test_data_utils.py`:

```python
from pathlib import Path

from data_utils import load_config


def test_load_config_has_required_sections():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    assert config["paths"]["data_dir"] == r"D:\paper\real_data"
    assert config["typical_days"]["enabled"] is True
    assert config["typical_days"]["k"] == 8
    assert config["technology"]["fitted"]["e_rdhx_kw_per_kw"] == 0.0265
    assert config["technology"]["fitted"]["cop_chiller"] == 3.0
    assert config["technology"]["fitted"]["ac_fan_kw_per_kw"] == 0.2137
```

- [ ] **Step 3: Run the failing config test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_data_utils.py::test_load_config_has_required_sections -v
```

Expected: FAIL because `data_utils.py` and `config.json` do not exist.

- [ ] **Step 4: Create minimal `config.json`**

Create `D:\paper\cooling_retrofit_code\config.json`:

```json
{
  "paths": {
    "data_dir": "D:\\paper\\real_data",
    "output_dir": "D:\\paper\\cooling_retrofit_code\\results"
  },
  "scenario": {
    "time_mode": "typical_days",
    "room_replication_factor": 1.0,
    "thermal_slack_penalty": 1000000.0
  },
  "typical_days": {
    "enabled": true,
    "k": 8,
    "random_seed": 42,
    "append_peak_day": true,
    "feature_columns": [
      "it_load_kw",
      "price_yuan_per_kwh",
      "carbon_kg_per_kwh",
      "outdoor_temp_c",
      "heating_demand_kw"
    ]
  },
  "solver": {
    "gurobi": {
      "time_limit_seconds": 120,
      "mip_gap": 0.01,
      "threads": 0
    },
    "nsga2": {
      "population_size": 24,
      "generations": 20,
      "crossover_probability": 0.9,
      "mutation_probability": 0.15,
      "random_seed": 42
    }
  },
  "technology": {
    "configurations": {
      "0": "existing_ac",
      "1": "new_ac",
      "2": "new_ac_with_ashp",
      "3": "new_ac_with_vent",
      "4": "cold_plate_cdu_wshp",
      "5": "rdhx_wshp",
      "6": "cold_plate_cdu_wshp_plus_air"
    },
    "fitted": {
      "e_rdhx_kw_per_kw": 0.0265,
      "cop_chiller": 3.0,
      "ac_fan_kw_per_kw": 0.2137
    },
    "cop_ashp": 3.2,
    "cop_wshp": 4.0,
    "tower_kw_per_kw_heat": 0.015,
    "cdu_pump_kw_per_kw": 0.02,
    "tes_loss_per_hour": 0.002,
    "bess_roundtrip_efficiency": 0.9
  },
  "economics": {
    "discount_rate": 0.06,
    "project_life_years": 15,
    "heat_credit_yuan_per_kwh": 0.2,
    "capex_yuan_per_kw": {
      "ac_new": 900,
      "vent": 200,
      "rdhx": 1200,
      "cdu": 800,
      "ashp": 700,
      "wshp": 900,
      "chiller": 1000,
      "tower": 250,
      "tes": 150,
      "bess_power": 600,
      "bess_energy": 900
    },
    "retrofit_fixed_yuan": {
      "0": 0,
      "1": 5000,
      "2": 6500,
      "3": 6200,
      "4": 9000,
      "5": 8500,
      "6": 10500
    }
  },
  "carbon": {
    "allow_negative_operational_carbon": false,
    "heat_displacement_kg_per_kwh": 0.11,
    "embodied_kg_per_kw": {
      "ac_new": 80,
      "vent": 20,
      "rdhx": 120,
      "cdu": 70,
      "ashp": 90,
      "wshp": 100,
      "chiller": 130,
      "tower": 25,
      "tes": 30,
      "bess_power": 70,
      "bess_energy": 120
    },
    "retrofit_fixed_kg": {
      "0": 0,
      "1": 400,
      "2": 520,
      "3": 500,
      "4": 700,
      "5": 650,
      "6": 820
    }
  },
  "weight": {
    "floor_limit_kg_per_m2": 800,
    "rack_static_kg": 900,
    "equipment_kg_per_kw": {
      "ac_new": 5,
      "rdhx": 6,
      "cdu": 5,
      "bess_energy": 10,
      "tes": 4
    }
  },
  "assumptions": {
    "economic_defaults": {
      "source": "assumed",
      "confidence": "medium",
      "description": "Used until project-specific vendor quotations are available."
    },
    "fitted_coefficients": {
      "source": "fitted",
      "confidence": "high",
      "description": "RDHX, chiller COP, and AC fan coefficients supplied by the researcher."
    }
  }
}
```

- [ ] **Step 5: Create minimal `data_utils.py`**

Add:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
```

- [ ] **Step 6: Run the config test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_data_utils.py::test_load_config_has_required_sections -v
```

Expected: PASS.

## Task 2: Real Data Loading And Alignment

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\data_utils.py`
- Modify: `D:\paper\cooling_retrofit_code\tests\test_data_utils.py`

- [ ] **Step 1: Add failing real-data tests**

Append:

```python
from data_utils import load_input_data


def test_load_input_data_reads_real_files():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    data = load_input_data(config)
    assert not data.rack_metadata.empty
    assert not data.heat_matrix.empty
    assert not data.time_series.empty
    assert not data.price.empty
    assert not data.carbon.empty
    assert not data.weather.empty
    assert not data.heating_demand.empty
    assert not data.ac_hourly.empty
    assert "timestamp_hour_utc" in data.ac_hourly.columns
    assert "cooling_strength_proxy_mean" in data.ac_hourly.columns


def test_input_data_exposes_hourly_frame():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    data = load_input_data(config)
    frame = data.hourly
    assert "timestamp_hour_utc" in frame.columns
    assert "it_load_kw" in frame.columns
    assert "price_yuan_per_kwh" in frame.columns
    assert "carbon_kg_per_kwh" in frame.columns
    assert "outdoor_temp_c" in frame.columns
    assert "heating_demand_kw" in frame.columns
```

- [ ] **Step 2: Run failing real-data tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_data_utils.py -v
```

Expected: FAIL because `load_input_data` is not implemented.

- [ ] **Step 3: Implement data containers and robust column picking**

Add to `data_utils.py`:

```python
from dataclasses import dataclass

import pandas as pd


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


def _read_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    return pd.read_csv(path)


def _timestamp_col(df: pd.DataFrame) -> str:
    for col in ("timestamp_hour_utc", "timestamp", "time", "datetime"):
        if col in df.columns:
            return col
    raise ValueError(f"No timestamp column found in columns: {list(df.columns)}")


def _numeric_col(df: pd.DataFrame, candidates: tuple[str, ...], fallback_contains: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    lower = {col: col.lower() for col in df.columns}
    for col, name in lower.items():
        if all(token in name for token in fallback_contains):
            return col
    numeric = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric:
        return numeric[-1]
    raise ValueError(f"No numeric column found in columns: {list(df.columns)}")


def _with_hour_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col = _timestamp_col(out)
    out["timestamp_hour_utc"] = pd.to_datetime(out[col], utc=True).dt.floor("h")
    return out
```

- [ ] **Step 4: Implement `load_input_data`**

Add:

```python
def load_input_data(config: dict[str, Any]) -> InputData:
    data_dir = Path(config["paths"]["data_dir"])
    rack_metadata = _read_csv(data_dir, "rack_metadata.csv")
    heat_matrix = _read_csv(data_dir, "H_matrix.csv")
    time_series = _with_hour_timestamp(_read_csv(data_dir, "time_series.csv"))
    price = _with_hour_timestamp(_read_csv(data_dir, "price_grid_beijing_2025_two_part_35_110kv_hourly_yuan_per_kwh.csv"))
    carbon = _with_hour_timestamp(_read_csv(data_dir, "carbon_factor_China_2025_kgCO2_per_kWh.csv"))
    weather = _with_hour_timestamp(_read_csv(data_dir, "weather_open_meteo_hourly.csv"))
    heating_demand = _with_hour_timestamp(_read_csv(data_dir, "heating_demand.csv"))
    ac_hourly = _with_hour_timestamp(_read_csv(data_dir, "m100_22-07_ac_unit_hourly.csv"))

    load_col = _numeric_col(time_series, ("it_load_kw", "power_kw", "load_kw"), ("load",))
    price_col = _numeric_col(price, ("price_yuan_per_kwh", "grid_price_yuan_per_kwh"), ("price",))
    carbon_col = _numeric_col(carbon, ("carbon_kg_per_kwh", "kgCO2_per_kWh", "kgco2_per_kwh"), ("carbon",))
    weather_col = _numeric_col(weather, ("outdoor_temp_c", "temperature_2m", "dry_bulb_temp_c"), ("temp",))
    heat_col = _numeric_col(heating_demand, ("heating_demand_kw", "heat_demand_kw"), ("heat",))

    hourly = (
        time_series.groupby("timestamp_hour_utc", as_index=False)[load_col].sum()
        .rename(columns={load_col: "it_load_kw"})
        .merge(price[["timestamp_hour_utc", price_col]].rename(columns={price_col: "price_yuan_per_kwh"}), on="timestamp_hour_utc", how="inner")
        .merge(carbon[["timestamp_hour_utc", carbon_col]].rename(columns={carbon_col: "carbon_kg_per_kwh"}), on="timestamp_hour_utc", how="inner")
        .merge(weather[["timestamp_hour_utc", weather_col]].rename(columns={weather_col: "outdoor_temp_c"}), on="timestamp_hour_utc", how="inner")
        .merge(heating_demand[["timestamp_hour_utc", heat_col]].rename(columns={heat_col: "heating_demand_kw"}), on="timestamp_hour_utc", how="inner")
        .sort_values("timestamp_hour_utc")
        .reset_index(drop=True)
    )
    if hourly.empty:
        raise ValueError("Aligned hourly input data is empty.")

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
    )
```

- [ ] **Step 5: Run data tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_data_utils.py -v
```

Expected: PASS. If a column name differs, update only `_numeric_col` candidates and rerun.

## Task 3: K-Means Representative Days

**Files:**
- Create: `D:\paper\cooling_retrofit_code\typical_days.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_typical_days.py`

- [ ] **Step 1: Write failing typical-day tests**

Create `tests\test_typical_days.py`:

```python
from pathlib import Path

from data_utils import load_config, load_input_data
from typical_days import build_time_index


def test_kmeans_typical_days_selects_real_days():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    data = load_input_data(config)
    result = build_time_index(data.hourly, config["typical_days"])
    assert result.mode == "typical_days"
    assert len(result.representative_days) == 8
    assert result.frame["representative_day_id"].nunique() == 8
    assert result.day_weights.sum() == data.hourly["timestamp_hour_utc"].dt.date.nunique()
    assert set(result.frame["timestamp_hour_utc"]).issubset(set(data.hourly["timestamp_hour_utc"]))


def test_full_timeseries_mode_returns_all_rows():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    config["typical_days"]["enabled"] = False
    data = load_input_data(config)
    result = build_time_index(data.hourly, config["typical_days"])
    assert result.mode == "full_timeseries"
    assert len(result.frame) == len(data.hourly)
    assert result.day_weights.sum() == data.hourly["timestamp_hour_utc"].dt.date.nunique()
```

- [ ] **Step 2: Run failing typical-day tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_typical_days.py -v
```

Expected: FAIL because `typical_days.py` does not exist.

- [ ] **Step 3: Implement dependency-light k-means**

Create `typical_days.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeIndex:
    mode: str
    frame: pd.DataFrame
    representative_days: list[pd.Timestamp]
    day_weights: pd.Series


def _daily_feature_matrix(hourly: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    frame = hourly.copy()
    frame["date"] = frame["timestamp_hour_utc"].dt.floor("D")
    complete_counts = frame.groupby("date").size()
    complete_days = complete_counts[complete_counts == 24].index
    frame = frame[frame["date"].isin(complete_days)].copy()
    features = []
    dates = []
    for date, day in frame.groupby("date", sort=True):
        values = []
        for col in feature_columns:
            if col in day.columns:
                values.extend(day[col].astype(float).to_numpy())
        if values:
            dates.append(date)
            features.append(values)
    matrix = np.asarray(features, dtype=float)
    if matrix.shape[0] == 0:
        raise ValueError("No complete days available for representative-day generation.")
    matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0))
    return pd.DataFrame({"date": dates}), matrix


def _kmeans(matrix: np.ndarray, k: int, seed: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if matrix.shape[0] < k:
        raise ValueError(f"Need at least {k} complete days, got {matrix.shape[0]}.")
    centroids = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = matrix[labels == idx]
            if len(members):
                centroids[idx] = members.mean(axis=0)
    return labels, centroids


def build_time_index(hourly: pd.DataFrame, typical_config: dict) -> TimeIndex:
    frame = hourly.copy()
    frame["timestamp_hour_utc"] = pd.to_datetime(frame["timestamp_hour_utc"], utc=True)
    frame["date"] = frame["timestamp_hour_utc"].dt.floor("D")
    all_day_count = frame["date"].nunique()
    if not typical_config.get("enabled", True):
        return TimeIndex(
            mode="full_timeseries",
            frame=frame.drop(columns=["date"]),
            representative_days=sorted(frame["date"].unique().tolist()),
            day_weights=pd.Series([1] * all_day_count),
        )

    k = int(typical_config["k"])
    feature_columns = list(typical_config["feature_columns"])
    day_index, matrix = _daily_feature_matrix(frame, feature_columns)
    labels, centroids = _kmeans(matrix, k=k, seed=int(typical_config.get("random_seed", 42)))
    selected_dates = []
    weights = []
    for cluster in range(k):
        member_idx = np.where(labels == cluster)[0]
        distances = np.linalg.norm(matrix[member_idx] - centroids[cluster], axis=1)
        chosen = member_idx[int(distances.argmin())]
        selected_dates.append(day_index.loc[chosen, "date"])
        weights.append(len(member_idx))
    selected_dates = sorted(pd.to_datetime(selected_dates, utc=True))
    typical = frame[frame["date"].isin(selected_dates)].copy()
    id_map = {date: idx for idx, date in enumerate(selected_dates)}
    weight_map = dict(zip(selected_dates, weights))
    typical["representative_day_id"] = typical["date"].map(id_map)
    typical["day_weight"] = typical["date"].map(weight_map)
    return TimeIndex(
        mode="typical_days",
        frame=typical.drop(columns=["date"]).reset_index(drop=True),
        representative_days=selected_dates,
        day_weights=pd.Series(weights),
    )
```

- [ ] **Step 4: Run typical-day tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_typical_days.py -v
```

Expected: PASS.

## Task 4: Decision Schema And Derived Configuration Flags

**Files:**
- Create: `D:\paper\cooling_retrofit_code\decision.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_decision_schema.py`

- [ ] **Step 1: Write failing decision tests**

Create `tests\test_decision_schema.py`:

```python
import numpy as np

from decision import DecisionSchema, PlanningDecision


def test_decode_configurations_and_derived_flags():
    schema = DecisionSchema(zone_ids=["z1", "z2"])
    vector = schema.encode_default()
    vector[0] = 4
    vector[1] = 5
    decision = schema.decode(vector)
    assert decision.s_z["z1"] == 4
    assert decision.s_z["z2"] == 5
    assert decision.p_z["z1"] == 1
    assert decision.r_z["z2"] == 1
    assert decision.w_z["z1"] == 1
    assert decision.w_z["z2"] == 1
    assert decision.a_z["z1"] == 0


def test_bounds_match_vector_length():
    schema = DecisionSchema(zone_ids=["z1", "z2", "z3"])
    lower, upper = schema.bounds()
    vector = schema.random_vector(np.random.default_rng(7))
    assert len(lower) == len(upper) == len(vector)
    assert all(l <= x <= u for l, x, u in zip(lower, vector, upper))
```

- [ ] **Step 2: Run failing decision tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_decision_schema.py -v
```

Expected: FAIL because `decision.py` does not exist.

- [ ] **Step 3: Implement `PlanningDecision` and `DecisionSchema`**

Create `decision.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanningDecision:
    zone_ids: list[str]
    s_z: dict[str, int]
    cap_ac_new: dict[str, float]
    cap_vent: dict[str, float]
    cap_rdhx: dict[str, float]
    cap_cdu: dict[str, float]
    cap_ashp: float
    cap_wshp: float
    cap_bess_e: float
    cap_bess_p: float
    cap_tes: float
    delta_cap_chiller: float
    delta_cap_tower: float
    o_z: dict[str, int]
    n_z: dict[str, int]
    p_z: dict[str, int]
    r_z: dict[str, int]
    a_z: dict[str, int]
    w_z: dict[str, int]


class DecisionSchema:
    zone_capacity_names = ("cap_ac_new", "cap_vent", "cap_rdhx", "cap_cdu")
    system_capacity_names = (
        "cap_ashp",
        "cap_wshp",
        "cap_bess_e",
        "cap_bess_p",
        "cap_tes",
        "delta_cap_chiller",
        "delta_cap_tower",
    )

    def __init__(self, zone_ids: list[str], zone_cap_upper_kw: float = 1000.0, system_cap_upper_kw: float = 5000.0):
        self.zone_ids = list(zone_ids)
        self.zone_cap_upper_kw = float(zone_cap_upper_kw)
        self.system_cap_upper_kw = float(system_cap_upper_kw)

    @property
    def vector_length(self) -> int:
        return len(self.zone_ids) + len(self.zone_ids) * len(self.zone_capacity_names) + len(self.system_capacity_names)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(self.vector_length, dtype=float)
        upper = np.zeros(self.vector_length, dtype=float)
        upper[: len(self.zone_ids)] = 6
        start = len(self.zone_ids)
        end = start + len(self.zone_ids) * len(self.zone_capacity_names)
        upper[start:end] = self.zone_cap_upper_kw
        upper[end:] = self.system_cap_upper_kw
        return lower, upper

    def encode_default(self) -> np.ndarray:
        return np.zeros(self.vector_length, dtype=float)

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        lower, upper = self.bounds()
        vector = rng.uniform(lower, upper)
        vector[: len(self.zone_ids)] = rng.integers(0, 7, size=len(self.zone_ids))
        return vector

    def decode(self, vector: np.ndarray) -> PlanningDecision:
        values = np.asarray(vector, dtype=float).copy()
        s_values = np.clip(np.rint(values[: len(self.zone_ids)]), 0, 6).astype(int)
        idx = len(self.zone_ids)
        zone_caps: dict[str, dict[str, float]] = {name: {} for name in self.zone_capacity_names}
        for name in self.zone_capacity_names:
            for zone in self.zone_ids:
                zone_caps[name][zone] = max(0.0, float(values[idx]))
                idx += 1
        system = {}
        for name in self.system_capacity_names:
            system[name] = max(0.0, float(values[idx]))
            idx += 1
        s_z = dict(zip(self.zone_ids, s_values.tolist()))
        o_z = {z: int(k == 0) for z, k in s_z.items()}
        n_z = {z: int(k in {1, 2, 3, 6}) for z, k in s_z.items()}
        p_z = {z: int(k in {4, 6}) for z, k in s_z.items()}
        r_z = {z: int(k == 5) for z, k in s_z.items()}
        a_z = {z: int(k == 2) for z, k in s_z.items()}
        w_z = {z: int(k in {4, 5, 6}) for z, k in s_z.items()}
        return PlanningDecision(
            zone_ids=self.zone_ids,
            s_z=s_z,
            cap_ac_new=zone_caps["cap_ac_new"],
            cap_vent=zone_caps["cap_vent"],
            cap_rdhx=zone_caps["cap_rdhx"],
            cap_cdu=zone_caps["cap_cdu"],
            o_z=o_z,
            n_z=n_z,
            p_z=p_z,
            r_z=r_z,
            a_z=a_z,
            w_z=w_z,
            **system,
        )
```

- [ ] **Step 4: Run decision tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_decision_schema.py -v
```

Expected: PASS.

## Task 5: Repair And Cheap Screening

**Files:**
- Create: `D:\paper\cooling_retrofit_code\repair.py`
- Create: `D:\paper\cooling_retrofit_code\feasibility_oracle.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_repair.py`

- [ ] **Step 1: Write failing repair tests**

Create `tests\test_repair.py`:

```python
import numpy as np

from decision import DecisionSchema
from feasibility_oracle import ScreeningResult
from repair import repair_vector, screen_decision


def test_repair_clears_incompatible_zone_capacities():
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 5
    vector[1] = 100
    vector[2] = 100
    vector[3] = 100
    vector[4] = 100
    repaired, actions = repair_vector(vector, schema, {"z1": 80}, {"z1": 200})
    decision = schema.decode(repaired)
    assert decision.cap_ac_new["z1"] == 0
    assert decision.cap_vent["z1"] == 0
    assert decision.cap_cdu["z1"] == 0
    assert decision.cap_rdhx["z1"] >= 80
    assert any("clear_ac_new" in action for action in actions)


def test_screen_blocks_peak_shortage():
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 4
    repaired, _ = repair_vector(vector, schema, {"z1": 100}, {"z1": 200})
    decision = schema.decode(repaired)
    result = screen_decision(decision, {"z1": 500}, {"z1": 200}, chiller_old_kw=0)
    assert isinstance(result, ScreeningResult)
    assert result.feasible is False
    assert result.violation > 0
```

- [ ] **Step 2: Run failing repair tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_repair.py -v
```

Expected: FAIL because repair modules do not exist.

- [ ] **Step 3: Implement screening result**

Create `feasibility_oracle.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScreeningResult:
    feasible: bool
    violation: float = 0.0
    reasons: list[str] = field(default_factory=list)
    skipped_lp_relaxation: bool = True


def run_lp_relaxation_screening() -> ScreeningResult:
    return ScreeningResult(feasible=True, reasons=["lp_relaxation_skipped"], skipped_lp_relaxation=True)
```

- [ ] **Step 4: Implement repair and screening**

Create `repair.py`:

```python
from __future__ import annotations

import numpy as np

from decision import DecisionSchema, PlanningDecision
from feasibility_oracle import ScreeningResult


def repair_vector(
    vector: np.ndarray,
    schema: DecisionSchema,
    peak_load_by_zone_kw: dict[str, float],
    floor_weight_margin_by_zone_kg: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    repaired = np.asarray(vector, dtype=float).copy()
    actions: list[str] = []
    lower, upper = schema.bounds()
    repaired = np.clip(repaired, lower, upper)
    repaired[: len(schema.zone_ids)] = np.clip(np.rint(repaired[: len(schema.zone_ids)]), 0, 6)
    idx = len(schema.zone_ids)
    offsets = {}
    for cap_name in schema.zone_capacity_names:
        offsets[cap_name] = {}
        for zone in schema.zone_ids:
            offsets[cap_name][zone] = idx
            idx += 1

    for zone_i, zone in enumerate(schema.zone_ids):
        config = int(repaired[zone_i])
        peak = float(peak_load_by_zone_kw.get(zone, 0.0))
        allowed = {
            "cap_ac_new": config in {1, 2, 3, 6},
            "cap_vent": config == 3,
            "cap_rdhx": config == 5,
            "cap_cdu": config in {4, 6},
        }
        for cap_name, is_allowed in allowed.items():
            pos = offsets[cap_name][zone]
            if not is_allowed and repaired[pos] != 0:
                repaired[pos] = 0
                actions.append(f"{zone}:clear_{cap_name}")
        if config in {1, 2, 3, 6} and repaired[offsets["cap_ac_new"][zone]] < peak:
            repaired[offsets["cap_ac_new"][zone]] = peak
            actions.append(f"{zone}:raise_ac_new_to_peak")
        if config == 5 and repaired[offsets["cap_rdhx"][zone]] < peak:
            repaired[offsets["cap_rdhx"][zone]] = peak
            actions.append(f"{zone}:raise_rdhx_to_peak")
        if config in {4, 6} and repaired[offsets["cap_cdu"][zone]] < peak:
            repaired[offsets["cap_cdu"][zone]] = peak
            actions.append(f"{zone}:raise_cdu_to_peak")

    return repaired, actions


def screen_decision(
    decision: PlanningDecision,
    peak_load_by_zone_kw: dict[str, float],
    floor_weight_margin_by_zone_kg: dict[str, float],
    chiller_old_kw: float,
) -> ScreeningResult:
    violation = 0.0
    reasons: list[str] = []
    chiller_load_bound = 0.0
    for zone in decision.zone_ids:
        peak = float(peak_load_by_zone_kw.get(zone, 0.0))
        supplied = (
            decision.cap_ac_new[zone]
            + decision.cap_vent[zone]
            + decision.cap_rdhx[zone]
            + decision.cap_cdu[zone]
        )
        if supplied + 1e-6 < peak:
            gap = peak - supplied
            violation += gap
            reasons.append(f"{zone}:peak_cooling_shortage={gap:.3f}")
        chiller_load_bound += decision.cap_ac_new[zone] + decision.cap_vent[zone] + decision.cap_rdhx[zone]
    chiller_capacity = chiller_old_kw + decision.delta_cap_chiller
    if chiller_capacity + 1e-6 < chiller_load_bound:
        gap = chiller_load_bound - chiller_capacity
        violation += gap
        reasons.append(f"chiller_peak_shortage={gap:.3f}")
    return ScreeningResult(feasible=violation <= 1e-6, violation=violation, reasons=reasons)
```

- [ ] **Step 5: Run repair tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_repair.py -v
```

Expected: PASS.

## Task 6: Inner Dispatch MILP Skeleton

**Files:**
- Create: `D:\paper\cooling_retrofit_code\inner_dispatch.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_inner_dispatch_small.py`

- [ ] **Step 1: Write failing inner-dispatch tests**

Create `tests\test_inner_dispatch_small.py`:

```python
import importlib.util

import numpy as np
import pandas as pd
import pytest

from decision import DecisionSchema
from inner_dispatch import InnerDispatchMILP


pytestmark = pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")


def test_inner_dispatch_solves_small_cdu_free_cooling_case():
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 4
    vector[4] = 100
    vector[-2] = 10
    vector[-1] = 150
    decision = schema.decode(vector)
    time_frame = pd.DataFrame({
        "timestamp_hour_utc": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
        "it_load_kw": [50.0, 60.0, 55.0],
        "price_yuan_per_kwh": [0.6, 0.7, 0.5],
        "carbon_kg_per_kwh": [0.5, 0.4, 0.45],
        "outdoor_temp_c": [5.0, 6.0, 7.0],
        "heating_demand_kw": [20.0, 20.0, 20.0],
        "day_weight": [1.0, 1.0, 1.0],
    })
    model = InnerDispatchMILP(config={"technology": {"cop_wshp": 4.0, "cop_ashp": 3.2, "tower_kw_per_kw_heat": 0.015, "cdu_pump_kw_per_kw": 0.02, "fitted": {"e_rdhx_kw_per_kw": 0.0265, "cop_chiller": 3.0, "ac_fan_kw_per_kw": 0.2137}}})
    result = model.solve(decision, time_frame, {"z1": 60.0}, chiller_old_kw=0.0, tower_old_kw=0.0)
    assert result.feasible is True
    assert result.dispatch["q_cdu_to_free_kw"].sum() >= 0
    assert result.dispatch["q_chiller_evap_kw"].sum() == pytest.approx(0.0, abs=1e-5)
```

- [ ] **Step 2: Run failing inner-dispatch test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_inner_dispatch_small.py -v
```

Expected: FAIL if `inner_dispatch.py` is absent; SKIPPED if `gurobipy` is unavailable.

- [ ] **Step 3: Implement dispatch dataclass and MILP**

Create `inner_dispatch.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InnerSolveResult:
    feasible: bool
    objective_value: float
    dispatch: pd.DataFrame
    diagnostics: dict[str, Any]


class InnerDispatchMILP:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def solve(self, decision, time_frame: pd.DataFrame, peak_load_by_zone_kw: dict[str, float], chiller_old_kw: float, tower_old_kw: float) -> InnerSolveResult:
        import gurobipy as gp
        from gurobipy import GRB

        tech = self.config["technology"]
        fitted = tech["fitted"]
        model = gp.Model("cooling_retrofit_dispatch")
        model.Params.OutputFlag = 0
        t_range = range(len(time_frame))
        q_air = model.addVars(t_range, lb=0.0, name="q_air")
        q_rdhx = model.addVars(t_range, lb=0.0, name="q_rdhx")
        q_cdu = model.addVars(t_range, lb=0.0, name="q_cdu")
        q_cdu_to_wshp = model.addVars(t_range, lb=0.0, name="q_cdu_to_wshp")
        q_cdu_to_free = model.addVars(t_range, lb=0.0, name="q_cdu_to_free")
        q_chiller = model.addVars(t_range, lb=0.0, name="q_chiller_evap")
        p_chiller = model.addVars(t_range, lb=0.0, name="p_chiller")
        p_tower = model.addVars(t_range, lb=0.0, name="p_tower")
        p_grid = model.addVars(t_range, lb=0.0, name="p_grid")
        q_heat = model.addVars(t_range, lb=0.0, name="q_heat")

        cap_air = sum(decision.cap_ac_new[z] + decision.cap_vent[z] for z in decision.zone_ids)
        cap_rdhx = sum(decision.cap_rdhx[z] for z in decision.zone_ids)
        cap_cdu = sum(decision.cap_cdu[z] for z in decision.zone_ids)
        cap_chiller = chiller_old_kw + decision.delta_cap_chiller
        cap_tower = tower_old_kw + decision.delta_cap_tower

        for t in t_range:
            load = float(time_frame.loc[t, "it_load_kw"])
            heat_demand = float(time_frame.loc[t, "heating_demand_kw"])
            model.addConstr(q_air[t] <= cap_air)
            model.addConstr(q_rdhx[t] <= cap_rdhx)
            model.addConstr(q_cdu[t] <= cap_cdu)
            model.addConstr(q_air[t] + q_rdhx[t] + q_cdu[t] >= load)
            model.addConstr(q_cdu_to_wshp[t] + q_cdu_to_free[t] == q_cdu[t])
            model.addConstr(q_chiller[t] == q_air[t] + q_rdhx[t])
            model.addConstr(q_chiller[t] <= cap_chiller)
            model.addConstr(p_chiller[t] == q_chiller[t] / float(fitted["cop_chiller"]))
            model.addConstr(q_heat[t] <= heat_demand)
            model.addConstr(q_heat[t] <= q_cdu_to_wshp[t] + q_rdhx[t])
            model.addConstr(q_chiller[t] + p_chiller[t] + q_cdu_to_free[t] <= cap_tower)
            model.addConstr(p_tower[t] == float(tech["tower_kw_per_kw_heat"]) * (q_chiller[t] + p_chiller[t] + q_cdu_to_free[t]))
            p_aux = fitted["e_rdhx_kw_per_kw"] * q_rdhx[t] + tech["cdu_pump_kw_per_kw"] * q_cdu[t]
            model.addConstr(p_grid[t] == p_chiller[t] + p_tower[t] + p_aux)

        model.setObjective(gp.quicksum(p_grid[t] * float(time_frame.loc[t, "price_yuan_per_kwh"]) for t in t_range), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            return InnerSolveResult(False, float("inf"), pd.DataFrame(), {"gurobi_status": model.Status})

        rows = []
        for t in t_range:
            rows.append({
                "timestamp_hour_utc": time_frame.loc[t, "timestamp_hour_utc"],
                "q_air_kw": q_air[t].X,
                "q_rdhx_kw": q_rdhx[t].X,
                "q_cdu_kw": q_cdu[t].X,
                "q_cdu_to_wshp_kw": q_cdu_to_wshp[t].X,
                "q_cdu_to_free_kw": q_cdu_to_free[t].X,
                "q_chiller_evap_kw": q_chiller[t].X,
                "p_chiller_kw": p_chiller[t].X,
                "p_tower_kw": p_tower[t].X,
                "p_grid_kw": p_grid[t].X,
                "q_heat_kw": q_heat[t].X,
                "day_weight": float(time_frame.loc[t].get("day_weight", 1.0)),
            })
        return InnerSolveResult(True, float(model.ObjVal), pd.DataFrame(rows), {"gurobi_status": model.Status})
```

- [ ] **Step 4: Run inner-dispatch test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_inner_dispatch_small.py -v
```

Expected: PASS with Gurobi available, SKIPPED if `gurobipy` is unavailable.

## Task 7: Evaluation Pipeline And Objectives

**Files:**
- Create: `D:\paper\cooling_retrofit_code\model.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_evaluate_solution.py`

- [ ] **Step 1: Write failing evaluation test**

Create `tests\test_evaluate_solution.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from data_utils import load_config, load_input_data
from decision import DecisionSchema
from model import evaluate_solution
from typical_days import build_time_index


pytestmark = pytest.mark.skipif(importlib.util.find_spec("gurobipy") is None, reason="gurobipy is not installed")


def test_evaluate_solution_returns_dual_objectives():
    config = load_config(Path(__file__).resolve().parents[1] / "config.json")
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 4
    vector[4] = float(time_index.frame["it_load_kw"].max())
    vector[-1] = float(time_index.frame["it_load_kw"].max()) * 2
    result = evaluate_solution(vector, schema, config, time_index.frame.head(24), {"z1": float(time_index.frame["it_load_kw"].max())})
    assert result.feasible is True
    assert result.tlcc >= 0
    assert result.tce >= 0
    assert "dispatch" in result.artifacts
```

- [ ] **Step 2: Run failing evaluation test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_evaluate_solution.py -v
```

Expected: FAIL because `model.py` does not exist; SKIPPED if Gurobi is unavailable.

- [ ] **Step 3: Implement evaluation result and cost/carbon aggregation**

Create `model.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from decision import DecisionSchema
from inner_dispatch import InnerDispatchMILP
from repair import repair_vector, screen_decision


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    tlcc: float
    tce: float
    violation: float = 0.0
    reasons: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def _crf(rate: float, years: int) -> float:
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def _capacity_cost(decision, config: dict[str, Any]) -> tuple[float, float]:
    capex = config["economics"]["capex_yuan_per_kw"]
    embodied = config["carbon"]["embodied_kg_per_kw"]
    rate = float(config["economics"]["discount_rate"])
    years = int(config["economics"]["project_life_years"])
    factor = _crf(rate, years)
    cost = 0.0
    carbon = 0.0
    for zone in decision.zone_ids:
        cost += decision.cap_ac_new[zone] * capex["ac_new"]
        cost += decision.cap_vent[zone] * capex["vent"]
        cost += decision.cap_rdhx[zone] * capex["rdhx"]
        cost += decision.cap_cdu[zone] * capex["cdu"]
        carbon += decision.cap_ac_new[zone] * embodied["ac_new"]
        carbon += decision.cap_vent[zone] * embodied["vent"]
        carbon += decision.cap_rdhx[zone] * embodied["rdhx"]
        carbon += decision.cap_cdu[zone] * embodied["cdu"]
    cost += decision.cap_ashp * capex["ashp"] + decision.cap_wshp * capex["wshp"]
    cost += decision.delta_cap_chiller * capex["chiller"] + decision.delta_cap_tower * capex["tower"]
    cost += decision.cap_tes * capex["tes"] + decision.cap_bess_p * capex["bess_power"] + decision.cap_bess_e * capex["bess_energy"]
    carbon += decision.cap_ashp * embodied["ashp"] + decision.cap_wshp * embodied["wshp"]
    carbon += decision.delta_cap_chiller * embodied["chiller"] + decision.delta_cap_tower * embodied["tower"]
    carbon += decision.cap_tes * embodied["tes"] + decision.cap_bess_p * embodied["bess_power"] + decision.cap_bess_e * embodied["bess_energy"]
    for zone in decision.zone_ids:
        k = str(decision.s_z[zone])
        cost += config["economics"]["retrofit_fixed_yuan"][k]
        carbon += config["carbon"]["retrofit_fixed_kg"][k]
    return cost * factor, carbon / years


def evaluate_solution(vector: np.ndarray, schema: DecisionSchema, config: dict[str, Any], time_frame: pd.DataFrame, peak_load_by_zone_kw: dict[str, float]) -> EvaluationResult:
    repaired, actions = repair_vector(vector, schema, peak_load_by_zone_kw, {z: 1e9 for z in schema.zone_ids})
    decision = schema.decode(repaired)
    screening = screen_decision(decision, peak_load_by_zone_kw, {z: 1e9 for z in schema.zone_ids}, chiller_old_kw=0.0)
    if not screening.feasible:
        return EvaluationResult(False, float("inf"), float("inf"), screening.violation, screening.reasons, {"repair_actions": actions})
    dispatch_result = InnerDispatchMILP(config).solve(decision, time_frame.reset_index(drop=True), peak_load_by_zone_kw, chiller_old_kw=0.0, tower_old_kw=0.0)
    if not dispatch_result.feasible:
        return EvaluationResult(False, float("inf"), float("inf"), 1e6, ["milp_infeasible"], {"diagnostics": dispatch_result.diagnostics})
    annualized_capex, annualized_embodied = _capacity_cost(decision, config)
    dispatch = dispatch_result.dispatch
    weighted_power = (dispatch["p_grid_kw"] * dispatch["day_weight"]).sum()
    weighted_heat = (dispatch["q_heat_kw"] * dispatch["day_weight"]).sum()
    price = time_frame["price_yuan_per_kwh"].mean()
    carbon_factor = time_frame["carbon_kg_per_kwh"].mean()
    operating_cost = weighted_power * price - weighted_heat * float(config["economics"]["heat_credit_yuan_per_kwh"])
    operating_carbon = weighted_power * carbon_factor - weighted_heat * float(config["carbon"]["heat_displacement_kg_per_kwh"])
    if not config["carbon"]["allow_negative_operational_carbon"]:
        operating_carbon = max(0.0, operating_carbon)
    return EvaluationResult(
        True,
        annualized_capex + operating_cost,
        annualized_embodied + operating_carbon,
        artifacts={"dispatch": dispatch, "repair_actions": actions, "decision": decision},
    )
```

- [ ] **Step 4: Run evaluation test**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_evaluate_solution.py -v
```

Expected: PASS with Gurobi available, SKIPPED if `gurobipy` is unavailable.

## Task 8: NSGA-II Core With Deb Constraint Handling

**Files:**
- Create: `D:\paper\cooling_retrofit_code\nsga2.py`
- Create: `D:\paper\cooling_retrofit_code\tests\test_nsga2.py`

- [ ] **Step 1: Write failing NSGA-II tests**

Create `tests\test_nsga2.py`:

```python
from model import EvaluationResult
from nsga2 import deb_better, fast_non_dominated_sort


def test_deb_prefers_feasible_over_infeasible():
    feasible = EvaluationResult(True, 10, 10)
    infeasible = EvaluationResult(False, 1, 1, violation=0.1)
    assert deb_better(feasible, infeasible) is True


def test_fast_non_dominated_sort_places_best_front_first():
    population = [
        EvaluationResult(True, 1, 5),
        EvaluationResult(True, 5, 1),
        EvaluationResult(True, 4, 4),
    ]
    fronts = fast_non_dominated_sort(population)
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]
```

- [ ] **Step 2: Run failing NSGA-II tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_nsga2.py -v
```

Expected: FAIL because `nsga2.py` does not exist.

- [ ] **Step 3: Implement comparison and sorting**

Create `nsga2.py`:

```python
from __future__ import annotations

import numpy as np


def dominates(a, b) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    return (a.tlcc <= b.tlcc and a.tce <= b.tce) and (a.tlcc < b.tlcc or a.tce < b.tce)


def deb_better(a, b) -> bool:
    return dominates(a, b)


def fast_non_dominated_sort(results: list) -> list[list[int]]:
    dominated_sets = [set() for _ in results]
    domination_counts = [0 for _ in results]
    fronts: list[list[int]] = [[]]
    for i, a in enumerate(results):
        for j, b in enumerate(results):
            if i == j:
                continue
            if dominates(a, b):
                dominated_sets[i].add(j)
            elif dominates(b, a):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            fronts[0].append(i)
    idx = 0
    while fronts[idx]:
        next_front = []
        for i in fronts[idx]:
            for j in dominated_sets[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        idx += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(front: list[int], results: list) -> dict[int, float]:
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}
    for attr in ("tlcc", "tce"):
        ordered = sorted(front, key=lambda idx: getattr(results[idx], attr))
        distance[ordered[0]] = distance[ordered[-1]] = float("inf")
        values = [getattr(results[idx], attr) for idx in ordered]
        span = max(values) - min(values)
        if span == 0:
            continue
        for pos in range(1, len(ordered) - 1):
            distance[ordered[pos]] += (values[pos + 1] - values[pos - 1]) / span
    return distance
```

- [ ] **Step 4: Run NSGA-II tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_nsga2.py -v
```

Expected: PASS.

## Task 9: Optimization Runner And Persistence

**Files:**
- Create: `D:\paper\cooling_retrofit_code\run_optimization.py`
- Create: `D:\paper\cooling_retrofit_code\main.py`

- [ ] **Step 1: Implement a small runner**

Create `run_optimization.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_config, load_input_data
from decision import DecisionSchema
from model import evaluate_solution
from nsga2 import fast_non_dominated_sort
from typical_days import build_time_index


def infer_zone_ids(data) -> list[str]:
    if "ac_unit" in data.ac_hourly.columns:
        return sorted(data.ac_hourly["ac_unit"].dropna().astype(str).unique().tolist())
    return ["z1"]


def run(config_path: str | Path) -> dict:
    config = load_config(config_path)
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])
    zone_ids = infer_zone_ids(data)
    schema = DecisionSchema(zone_ids)
    rng = np.random.default_rng(config["solver"]["nsga2"]["random_seed"])
    peak = float(time_index.frame["it_load_kw"].max()) / max(1, len(zone_ids))
    peak_by_zone = {zone: peak for zone in zone_ids}
    population_size = int(config["solver"]["nsga2"]["population_size"])
    population = [schema.random_vector(rng) for _ in range(population_size)]
    results = [evaluate_solution(vec, schema, config, time_index.frame, peak_by_zone) for vec in population]
    fronts = fast_non_dominated_sort(results)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, result in enumerate(results):
        rows.append({
            "solution_id": idx,
            "feasible": result.feasible,
            "tlcc": result.tlcc,
            "tce": result.tce,
            "violation": result.violation,
            "front": next((rank for rank, front in enumerate(fronts) if idx in front), -1),
            "reasons": ";".join(result.reasons),
        })
    all_eval = pd.DataFrame(rows)
    all_eval.to_csv(output_dir / "all_evaluations.csv", index=False, encoding="utf-8-sig")
    pareto = all_eval[all_eval["front"] == 0].copy()
    pareto.to_csv(output_dir / "pareto_solutions.csv", index=False, encoding="utf-8-sig")
    return {"config": config, "all_evaluations": all_eval, "pareto": pareto}
```

- [ ] **Step 2: Implement CLI**

Create `main.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from run_optimization import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cooling retrofit optimization.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    args = parser.parse_args()
    result = run(args.config)
    print(f"Saved {len(result['all_evaluations'])} evaluations.")
    print(f"Pareto solutions: {len(result['pareto'])}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run runner smoke command**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python main.py --config config.json
```

Expected with Gurobi: CSV files appear in `D:\paper\cooling_retrofit_code\results`. Expected without Gurobi: command raises `ModuleNotFoundError: No module named 'gurobipy'`; install or activate Gurobi before full optimization.

## Task 10: Reporting And Plots

**Files:**
- Create: `D:\paper\cooling_retrofit_code\plotting.py`
- Create: `D:\paper\cooling_retrofit_code\analyze_results.py`
- Modify: `D:\paper\cooling_retrofit_code\run_optimization.py`

- [ ] **Step 1: Implement plotting**

Create `plotting.py`:

```python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_pareto_plot(pareto: pd.DataFrame, output_dir: str | Path) -> Path:
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "pareto.png"
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(pareto["tlcc"], pareto["tce"], s=28)
    ax.set_xlabel("TLCC (yuan/year)")
    ax.set_ylabel("TCE (kgCO2/year)")
    ax.set_title("Pareto Front")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
```

- [ ] **Step 2: Implement Chinese Markdown report**

Create `analyze_results.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report(all_evaluations: pd.DataFrame, pareto: pd.DataFrame, config: dict, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.md"
    feasible_count = int(all_evaluations["feasible"].sum())
    best_cost = pareto["tlcc"].min() if not pareto.empty else float("nan")
    best_carbon = pareto["tce"].min() if not pareto.empty else float("nan")
    lines = [
        "# 冷却改造优化结果报告",
        "",
        f"- 总评估方案数：{len(all_evaluations)}",
        f"- 可行方案数：{feasible_count}",
        f"- Pareto 方案数：{len(pareto)}",
        f"- Pareto 最低年化成本：{best_cost:.3f}",
        f"- Pareto 最低年化碳排：{best_carbon:.3f}",
        "",
        "## 参数来源",
        "",
    ]
    for name, meta in config["assumptions"].items():
        lines.append(f"- {name}：{meta['source']}，置信度 {meta['confidence']}，{meta['description']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 3: Connect reporting in runner**

Modify `run_optimization.py` by importing and calling:

```python
from analyze_results import write_report
from plotting import save_pareto_plot
```

Then after writing `pareto_solutions.csv`:

```python
if not pareto.empty:
    save_pareto_plot(pareto, output_dir)
write_report(all_eval, pareto, config, output_dir)
```

- [ ] **Step 4: Run report smoke command**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python main.py --config config.json
```

Expected with Gurobi: `results\report.md` and `results\figures\pareto.png` exist.

## Task 11: Documentation

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\README.md`
- Create: `D:\paper\cooling_retrofit_code\docs\model_mapping.md`
- Create: `D:\paper\cooling_retrofit_code\docs\assumptions.md`

- [ ] **Step 1: Write README**

Create:

```markdown
# Cooling Retrofit Code

This project implements the first runnable codebase for the cooling retrofit planning model.

## Data

Default input directory: `D:\paper\real_data`.

Required files:

- `rack_metadata.csv`
- `H_matrix.csv`
- `time_series.csv`
- `price_grid_beijing_2025_two_part_35_110kv_hourly_yuan_per_kwh.csv`
- `carbon_factor_China_2025_kgCO2_per_kWh.csv`
- `weather_open_meteo_hourly.csv`
- `heating_demand.csv`
- `m100_22-07_ac_unit_hourly.csv`

## Run

```powershell
cd "D:\paper\cooling_retrofit_code"
python main.py --config config.json
```

## Test

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests -v
```

Gurobi tests are skipped when `gurobipy` is not installed.
```

- [ ] **Step 2: Write model mapping**

Create `docs\model_mapping.md`:

```markdown
# Model Mapping

| Paper symbol | Code field | Meaning |
| --- | --- | --- |
| `s_z` | `PlanningDecision.s_z` | Retrofit configuration code for zone `z` |
| `u_{z,k}` | derived from `s_z` | One-hot retrofit choice |
| `Cap_z^{AC,new}` | `cap_ac_new[z]` | New air-conditioner capacity |
| `Cap_z^{vent}` | `cap_vent[z]` | Auxiliary ventilation capacity |
| `Cap_z^{RDHX}` | `cap_rdhx[z]` | RDHX capacity |
| `Cap_z^{CDU}` | `cap_cdu[z]` | CDU capacity for cold plate configurations |
| `P_t^{Chiller}` | `p_chiller_kw` | Chiller electric power |
| `Q_t^{Tower}` | implicit tower heat expression | Chiller condenser heat plus CDU direct rejection |
| `s_{z,t}^{air}` | `q_air_kw` | Effective air-side cooling, bounded by air-side capacity |
| `TLCC` | `EvaluationResult.tlcc` | Annualized total life-cycle cost |
| `TCE` | `EvaluationResult.tce` | Annualized total carbon emission |
```

- [ ] **Step 3: Write assumptions document**

Create `docs\assumptions.md`:

```markdown
# Assumptions

The following fitted coefficients are supplied by the researcher:

- RDHX overall electric coefficient: `0.0265 kWe/kWth`
- Chiller COP: `3.0`
- AC terminal fan coefficient: `0.2137 kWe/kWth`

Economic, embodied-carbon, equipment-weight, and default COP values in `config.json` are assumed placeholders for model execution and must be replaced or cited before final paper experiments.

Fixed retrofit cost and fixed retrofit embodied carbon represent construction, interface, commissioning, downtime, and retrofit work only. They exclude capacity equipment CAPEX and embodied carbon to avoid double counting.
```

- [ ] **Step 4: Read docs for consistency**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
Get-Content README.md
Get-Content docs\model_mapping.md
Get-Content docs\assumptions.md
```

Expected: all three files are readable and contain the data path, fitted coefficients, and double-counting rule.

## Task 12: Final Verification

**Files:**
- All project files under `D:\paper\cooling_retrofit_code\`.

- [ ] **Step 1: Run non-MILP tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests\test_data_utils.py tests\test_typical_days.py tests\test_decision_schema.py tests\test_repair.py tests\test_nsga2.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
python -m pytest tests -v
```

Expected: PASS if Gurobi is available; otherwise PASS with Gurobi-dependent tests SKIPPED.

- [ ] **Step 3: Scan for old-model residue**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
Select-String -Path *.py,tests\*.py -Pattern "y_i|x_ig|n_crac_remove|CRAC|设备组液冷" -SimpleMatch
```

Expected: no matches.

- [ ] **Step 4: Verify required outputs after smoke run**

Run:

```powershell
cd "D:\paper\cooling_retrofit_code"
Test-Path results\all_evaluations.csv
Test-Path results\pareto_solutions.csv
Test-Path results\report.md
```

Expected with Gurobi smoke run: all three commands print `True`.

## Self-Review

- Spec coverage: the plan creates the target project, reads `D:\paper\real_data`, uses K=8 representative days, implements seven configuration decoding, adds `repair.py`, adds cheap screening, keeps LP relaxation skipped, builds Gurobi inner dispatch, aggregates TLCC/TCE, and writes CSV/figure/Chinese report outputs.
- Placeholder scan: no unresolved placeholders are present in executable steps. Assumed economic values are explicit config defaults and documented as assumptions.
- Type consistency: `InputData`, `TimeIndex`, `PlanningDecision`, `ScreeningResult`, `InnerSolveResult`, and `EvaluationResult` are introduced before downstream modules import them.
- Domain consistency: cold plate/CDU heat bypasses Chiller, RDHX and air-side loads use Chiller, tower handles Chiller condenser heat plus CDU direct rejection, and RDHX power uses a single fitted coefficient.
