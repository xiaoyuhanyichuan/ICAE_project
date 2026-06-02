# Room Spatial Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add static room-layout and growth-aging relationship artifacts for the Pareto knee solution, then include them in the result report.

**Architecture:** Add a focused `spatial_analysis.py` module that selects the Pareto knee solution, builds rack-level and zone-level analysis frames, writes two CSV files, and saves two PNG figures. Integrate it after Pareto export in `run_optimization.py`; update `analyze_results.py` so the report indexes and interprets the spatial artifacts when they exist.

**Tech Stack:** Python, pandas, numpy, matplotlib, pytest; existing project files under `D:\paper\cooling_retrofit_code`.

---

## File Structure

- Create `D:\paper\cooling_retrofit_code\spatial_analysis.py`
  - Owns spatial data preparation, knee-solution selection, plotting, artifact writing, and compact summary generation.
- Create `D:\paper\cooling_retrofit_code\tests\test_spatial_analysis.py`
  - Tests knee selection, rack-to-config mapping, parameter fallback, zone aggregation, and artifact writing with temporary paths.
- Modify `D:\paper\cooling_retrofit_code\run_optimization.py`
  - Calls `save_spatial_analysis_artifacts(...)` after Pareto plots and before report generation.
  - Returns `spatial_result` in the `run(...)` dictionary.
- Modify `D:\paper\cooling_retrofit_code\analyze_results.py`
  - Reads spatial CSV/PNG artifacts from `output_dir`.
  - Adds a “机房空间与构型解释” section before the output index.
  - Adds the two spatial figures and two spatial CSV files to the output index.
- Modify `D:\paper\cooling_retrofit_code\config.json`
  - Adds explicit model-input defaults for `rho_load` and `alpha_age`.
  - Keeps the values marked as scenario/fallback inputs until measured values are supplied.

The project directory is not currently a git repository, so commit steps are intentionally omitted.

---

### Task 1: Add Tests for Spatial Analysis Core

**Files:**
- Create: `D:\paper\cooling_retrofit_code\tests\test_spatial_analysis.py`
- Target module to be created later: `D:\paper\cooling_retrofit_code\spatial_analysis.py`

- [ ] **Step 1: Write the failing test file**

Create `tests\test_spatial_analysis.py` with these tests:

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_analysis import (
    build_rack_layout_frame,
    build_zone_relationship_frame,
    config_name_map,
    run_spatial_analysis,
    select_knee_solution,
)


def _config():
    return {
        "technology": {
            "configurations": {
                "0": "existing_ac",
                "4": "cold_plate_cdu_wshp",
                "5": "rdhx_wshp",
            }
        },
        "scenario": {
            "business_growth": {
                "default_rho_load": 0.08,
                "rho_load_by_ac_unit": {"cdz1": 0.12},
                "rho_load_by_rack": {"rack_it_002": 0.20},
            },
            "ac_aging": {
                "default_alpha_age": 0.10,
                "alpha_age_by_ac_unit": {"cdz1": 0.25, "cdz2": 0.05},
            },
        },
    }


def _rack_metadata():
    return pd.DataFrame(
        {
            "rack_id": ["rack_it_000", "rack_it_001", "rack_it_002"],
            "row": [2, 2, 6],
            "col": [21, 20, 5],
            "ac_unit": ["cdz1", "cdz1", "cdz2"],
            "hotspot_risk": [0.1, 0.4, 0.9],
        }
    )


def _hourly():
    return pd.DataFrame(
        {
            "rack_it_000_kw": [10.0, 30.0],
            "rack_it_001_kw": [20.0, 10.0],
            "rack_it_002_kw": [5.0, 50.0],
        }
    )


def _pareto():
    return pd.DataFrame(
        {
            "solution_id": [1, 2, 3],
            "tlcc": [100.0, 150.0, 120.0],
            "tce": [80.0, 40.0, 60.0],
            "config_by_zone_json": [
                json.dumps({"cdz1": 0, "cdz2": 0}),
                json.dumps({"cdz1": 5, "cdz2": 4}),
                json.dumps({"cdz1": 4, "cdz2": 5}),
            ],
        }
    )


def test_select_knee_solution_uses_normalized_ideal_distance():
    row = select_knee_solution(_pareto())

    assert int(row["solution_id"]) == 3


def test_build_rack_layout_frame_maps_config_growth_and_age():
    row = select_knee_solution(_pareto())

    frame = build_rack_layout_frame(row, _rack_metadata(), _config())

    assert frame["solution_id"].tolist() == [3, 3, 3]
    assert frame["config_id"].tolist() == [4, 4, 5]
    assert frame["config_name"].tolist() == ["cold_plate_cdu_wshp", "cold_plate_cdu_wshp", "rdhx_wshp"]
    assert frame["rho_load"].tolist() == [0.12, 0.12, 0.20]
    assert frame["alpha_age"].tolist() == [0.25, 0.25, 0.05]


def test_build_zone_relationship_frame_aggregates_peak_load_and_hotspot():
    row = select_knee_solution(_pareto())
    rack_frame = build_rack_layout_frame(row, _rack_metadata(), _config())

    frame = build_zone_relationship_frame(row, rack_frame, _hourly(), _config())

    cdz1 = frame.set_index("ac_unit").loc["cdz1"]
    cdz2 = frame.set_index("ac_unit").loc["cdz2"]
    assert float(cdz1["rho_load_mean"]) == 0.12
    assert float(cdz1["alpha_age"]) == 0.25
    assert float(cdz1["zone_peak_it_kw"]) == 50.0
    assert float(cdz1["zone_max_hotspot_risk"]) == 0.4
    assert float(cdz2["rho_load_mean"]) == 0.20
    assert float(cdz2["zone_peak_it_kw"]) == 50.0


def test_run_spatial_analysis_writes_expected_artifacts(tmp_path):
    result = run_spatial_analysis(
        pareto=_pareto(),
        rack_metadata=_rack_metadata(),
        hourly=_hourly(),
        config=_config(),
        output_dir=tmp_path,
    )

    assert result["available"] is True
    assert Path(result["room_layout_path"]).name == "room_layout_knee_solution.png"
    assert Path(result["relationship_path"]).name == "config_choice_vs_growth_age.png"
    assert Path(result["rack_layout_csv"]).exists()
    assert Path(result["relationship_csv"]).exists()
    assert Path(result["room_layout_path"]).exists()
    assert Path(result["relationship_path"]).exists()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```powershell
python -m pytest tests\test_spatial_analysis.py -q -p no:cacheprovider
```

Expected: failure with `ModuleNotFoundError: No module named 'spatial_analysis'`.

---

### Task 2: Implement `spatial_analysis.py`

**Files:**
- Create: `D:\paper\cooling_retrofit_code\spatial_analysis.py`
- Test: `D:\paper\cooling_retrofit_code\tests\test_spatial_analysis.py`

- [ ] **Step 1: Add the module with data-frame builders and plotting**

Create `spatial_analysis.py` with:

```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def config_name_map(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("technology", {}).get("configurations", {})
    return {int(key): str(value) for key, value in raw.items()}


def select_knee_solution(pareto: pd.DataFrame) -> pd.Series:
    if pareto.empty:
        raise ValueError("Cannot select knee solution from an empty Pareto frame.")
    finite = pareto.copy()
    finite["tlcc_num"] = pd.to_numeric(finite["tlcc"], errors="coerce")
    finite["tce_num"] = pd.to_numeric(finite["tce"], errors="coerce")
    finite = finite[np.isfinite(finite["tlcc_num"]) & np.isfinite(finite["tce_num"])]
    if finite.empty:
        raise ValueError("Cannot select knee solution without finite TLCC/TCE values.")
    if len(finite) == 1:
        return finite.iloc[0]
    tlcc_span = finite["tlcc_num"].max() - finite["tlcc_num"].min()
    tce_span = finite["tce_num"].max() - finite["tce_num"].min()
    norm_cost = pd.Series(0.0, index=finite.index) if tlcc_span <= 0 else (finite["tlcc_num"] - finite["tlcc_num"].min()) / tlcc_span
    norm_carbon = pd.Series(0.0, index=finite.index) if tce_span <= 0 else (finite["tce_num"] - finite["tce_num"].min()) / tce_span
    knee_idx = (norm_cost.pow(2) + norm_carbon.pow(2)).idxmin()
    return finite.loc[knee_idx]


def _growth_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("scenario", {}).get("business_growth", {})


def _aging_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("scenario", {}).get("ac_aging", {})


def _rho_load_for_rack(rack_id: str, ac_unit: str, config: dict[str, Any]) -> float:
    growth = _growth_config(config)
    by_rack = growth.get("rho_load_by_rack", {})
    by_ac = growth.get("rho_load_by_ac_unit", {})
    default = _to_float(growth.get("default_rho_load", 0.0))
    if rack_id in by_rack:
        return _to_float(by_rack[rack_id], default)
    if ac_unit in by_ac:
        return _to_float(by_ac[ac_unit], default)
    return default


def _alpha_age_for_ac(ac_unit: str, config: dict[str, Any]) -> float:
    aging = _aging_config(config)
    by_ac = aging.get("alpha_age_by_ac_unit", {})
    default = _to_float(aging.get("default_alpha_age", 0.0))
    return _to_float(by_ac.get(ac_unit, default), default)


def _rack_label(rack_id: str) -> str:
    match = re.search(r"(\d+)$", str(rack_id))
    return str(int(match.group(1))) if match else str(rack_id)


def build_rack_layout_frame(
    knee_row: pd.Series,
    rack_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = {"rack_id", "row", "col", "ac_unit"}
    missing = sorted(required - set(rack_metadata.columns))
    if missing:
        raise ValueError(f"rack_metadata is missing required columns: {missing}")
    config_by_zone = _safe_json(knee_row.get("config_by_zone_json"))
    if not isinstance(config_by_zone, dict):
        raise ValueError("knee solution is missing config_by_zone_json.")
    names = config_name_map(config)
    frame = rack_metadata.copy()
    frame["solution_id"] = int(knee_row.get("solution_id", -1))
    frame["ac_unit"] = frame["ac_unit"].astype(str)
    frame["config_id"] = frame["ac_unit"].map(lambda zone: int(config_by_zone.get(zone, -1)))
    frame["config_name"] = frame["config_id"].map(lambda item: names.get(int(item), f"config_{int(item)}"))
    frame["rho_load"] = frame.apply(lambda row: _rho_load_for_rack(str(row["rack_id"]), str(row["ac_unit"]), config), axis=1)
    frame["alpha_age"] = frame["ac_unit"].map(lambda zone: _alpha_age_for_ac(str(zone), config))
    frame["rack_label"] = frame["rack_id"].map(_rack_label)
    if "hotspot_risk" not in frame.columns:
        frame["hotspot_risk"] = np.nan
    return frame[
        [
            "solution_id",
            "rack_id",
            "rack_label",
            "row",
            "col",
            "ac_unit",
            "config_id",
            "config_name",
            "rho_load",
            "alpha_age",
            "hotspot_risk",
        ]
    ]


def build_zone_relationship_frame(
    knee_row: pd.Series,
    rack_frame: pd.DataFrame,
    hourly: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    del config
    rows: list[dict[str, Any]] = []
    solution_id = int(knee_row.get("solution_id", -1))
    for ac_unit, group in rack_frame.groupby("ac_unit", sort=True):
        rack_cols = [f"{rack_id}_kw" for rack_id in group["rack_id"].astype(str) if f"{rack_id}_kw" in hourly.columns]
        if rack_cols:
            zone_series = hourly[rack_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            peak_it = float(zone_series.max())
        else:
            peak_it = 0.0
        first = group.iloc[0]
        rows.append(
            {
                "solution_id": solution_id,
                "ac_unit": ac_unit,
                "config_id": int(first["config_id"]),
                "config_name": str(first["config_name"]),
                "rack_count": int(len(group)),
                "rho_load_mean": float(pd.to_numeric(group["rho_load"], errors="coerce").mean()),
                "alpha_age": float(pd.to_numeric(group["alpha_age"], errors="coerce").mean()),
                "zone_peak_it_kw": peak_it,
                "zone_mean_hotspot_risk": float(pd.to_numeric(group["hotspot_risk"], errors="coerce").mean()),
                "zone_max_hotspot_risk": float(pd.to_numeric(group["hotspot_risk"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)
```

Then add plotting and orchestration functions in the same file:

```python
def _color_map(values: pd.Series) -> dict[int, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    unique = sorted({int(value) for value in values.dropna().tolist()})
    cmap = plt.get_cmap("tab10")
    return {value: cmap(index % 10) for index, value in enumerate(unique)}


def save_room_layout_plot(rack_frame: pd.DataFrame, knee_row: pd.Series, output_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "room_layout_knee_solution.png"
    colors = _color_map(rack_frame["config_id"])

    fig, ax = plt.subplots(figsize=(13, 6.5))
    for config_id, group in rack_frame.groupby("config_id", sort=True):
        label = f"{int(config_id)} {group['config_name'].iloc[0]}"
        edgecolors = [
            "red" if _to_float(value, 0.0) >= 0.8 else "black"
            for value in group["hotspot_risk"].tolist()
        ]
        ax.scatter(
            pd.to_numeric(group["col"], errors="coerce"),
            pd.to_numeric(group["row"], errors="coerce"),
            s=420,
            color=colors[int(config_id)],
            edgecolors=edgecolors,
            linewidths=1.8,
            alpha=0.86,
            label=label,
        )
        for _, row in group.iterrows():
            ax.text(float(row["col"]), float(row["row"]), str(row["rack_label"]), ha="center", va="center", fontsize=9)
            ax.text(float(row["col"]), float(row["row"]) + 0.24, str(row["ac_unit"]), ha="center", va="bottom", fontsize=6, color="#333333")

    tlcc = _to_float(knee_row.get("tlcc"), float("nan"))
    tce = _to_float(knee_row.get("tce"), float("nan"))
    ax.set_title(f"Room layout retrofit configuration - knee solution {int(knee_row.get('solution_id', -1))}\\nTLCC={tlcc:,.0f} yuan/year, TCE={tce:,.0f} kgCO2/year")
    ax.set_xlabel("Rack column")
    ax.set_ylabel("Rack row")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.invert_yaxis()
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Configuration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_growth_age_plot(zone_frame: pd.DataFrame, output_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "config_choice_vs_growth_age.png"
    colors = _color_map(zone_frame["config_id"])
    max_peak = max(float(pd.to_numeric(zone_frame["zone_peak_it_kw"], errors="coerce").max()), 1.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    for config_id, group in zone_frame.groupby("config_id", sort=True):
        sizes = 180.0 + 520.0 * pd.to_numeric(group["zone_peak_it_kw"], errors="coerce").fillna(0.0) / max_peak
        ax.scatter(
            pd.to_numeric(group["rho_load_mean"], errors="coerce"),
            pd.to_numeric(group["alpha_age"], errors="coerce"),
            s=sizes,
            color=colors[int(config_id)],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.82,
            label=f"{int(config_id)} {group['config_name'].iloc[0]}",
        )
        for _, row in group.iterrows():
            ax.text(float(row["rho_load_mean"]), float(row["alpha_age"]), str(row["ac_unit"]), ha="center", va="center", fontsize=8)

    ax.set_title("Configuration choice vs. business growth and AC aging")
    ax.set_xlabel("Zone mean business load growth intensity rho_load")
    ax.set_ylabel("Old AC capacity aging rate alpha_age")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", title="Configuration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_spatial_analysis(
    pareto: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    hourly: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if pareto.empty:
        return {"available": False, "reason": "empty_pareto"}
    try:
        knee = select_knee_solution(pareto)
        rack_frame = build_rack_layout_frame(knee, rack_metadata, config)
        zone_frame = build_zone_relationship_frame(knee, rack_frame, hourly, config)
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}

    spatial_dir = Path(output_dir) / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    rack_csv = spatial_dir / "knee_solution_rack_layout.csv"
    relationship_csv = spatial_dir / "config_choice_relationship.csv"
    rack_frame.to_csv(rack_csv, index=False, encoding="utf-8-sig")
    zone_frame.to_csv(relationship_csv, index=False, encoding="utf-8-sig")
    room_plot = save_room_layout_plot(rack_frame, knee, output_dir)
    relationship_plot = save_growth_age_plot(zone_frame, output_dir)

    return {
        "available": True,
        "solution_id": int(knee.get("solution_id", -1)),
        "room_layout_path": str(room_plot),
        "relationship_path": str(relationship_plot),
        "rack_layout_csv": str(rack_csv),
        "relationship_csv": str(relationship_csv),
        "zone_count": int(len(zone_frame)),
        "rack_count": int(len(rack_frame)),
    }
```

- [ ] **Step 2: Run the spatial analysis tests**

Run:

```powershell
python -m pytest tests\test_spatial_analysis.py -q -p no:cacheprovider
```

Expected: all tests pass.

---

### Task 3: Add Explicit Growth and Aging Parameters to Configuration

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\config.json`
- Test: `D:\paper\cooling_retrofit_code\tests\test_spatial_analysis.py`

- [ ] **Step 1: Add scenario input blocks**

In `config.json`, under the existing `"scenario"` object, add:

```json
"business_growth": {
  "default_rho_load": 0.08,
  "rho_load_by_ac_unit": {
    "cdz1": 0.08,
    "cdz2": 0.10,
    "cdz3": 0.12,
    "cdz4": 0.15,
    "cdz5": 0.18,
    "cdz6": 0.20
  },
  "source": "Scenario fallback for model Section 5.1 rho_i_load; replace with measured or forecast rack-level business growth inputs."
},
"ac_aging": {
  "default_alpha_age": 0.12,
  "alpha_age_by_ac_unit": {
    "cdz1": 0.08,
    "cdz2": 0.10,
    "cdz3": 0.12,
    "cdz4": 0.16,
    "cdz5": 0.20,
    "cdz6": 0.24
  },
  "source": "Scenario fallback for model Section 5.2 alpha_z_age; replace with AC-unit health or service-age data."
}
```

Keep valid JSON commas around neighboring blocks.

- [ ] **Step 2: Validate JSON**

Run:

```powershell
python -m json.tool config.json > $null
```

Expected: no output and exit code 0.

- [ ] **Step 3: Run spatial analysis tests again**

Run:

```powershell
python -m pytest tests\test_spatial_analysis.py -q -p no:cacheprovider
```

Expected: all tests pass.

---

### Task 4: Integrate Spatial Artifacts into Optimization Run

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\run_optimization.py`
- Test: `D:\paper\cooling_retrofit_code\tests\test_reporting.py`

- [ ] **Step 1: Write a failing run integration test**

Add this test to `tests\test_reporting.py`:

```python
def test_run_calls_spatial_analysis_for_non_empty_pareto(monkeypatch):
    config = {
        "paths": {"output_dir": "out"},
        "solver": {"nsga2": {"population_size": 1, "random_seed": 42}},
        "typical_days": {},
        "assumptions": {},
    }
    data = SimpleNamespace(
        ac_hourly=pd.DataFrame({"ac_unit": ["z1"]}),
        rack_metadata=pd.DataFrame({"rack_id": ["rack_it_000"], "row": [1], "col": [1], "ac_unit": ["z1"]}),
        hourly=pd.DataFrame({"it_load_kw": [10.0], "rack_it_000_kw": [10.0]}),
    )
    time_index = SimpleNamespace(frame=data.hourly)

    class FakeSchema:
        def __init__(self, zone_ids):
            self.zone_ids = zone_ids

        def random_vector(self, rng):
            return np.array([1.0])

    decision = SimpleNamespace(
        s_z={"z1": 0},
        zone_ids=["z1"],
        cap_ac_new={"z1": 0.0},
        cap_vent={"z1": 0.0},
        cap_rdhx={"z1": 0.0},
        cap_cdu={"z1": 0.0},
        cap_ashp=0.0,
        cap_wshp=0.0,
        cap_bess_e=0.0,
        cap_bess_p=0.0,
        cap_tes=0.0,
        delta_cap_chiller=0.0,
        delta_cap_tower=0.0,
    )
    result = EvaluationResult(True, 100.0, 80.0, artifacts={"decision": decision})
    spatial_calls = []

    monkeypatch.setattr(run_optimization, "load_config", lambda path: config)
    monkeypatch.setattr(run_optimization, "load_input_data", lambda loaded_config: data)
    monkeypatch.setattr(run_optimization, "build_time_index", lambda hourly, typical_days: time_index)
    monkeypatch.setattr(run_optimization, "DecisionSchema", FakeSchema)
    monkeypatch.setattr(run_optimization, "evaluate_solution", lambda *args, **kwargs: result)
    monkeypatch.setattr(run_optimization, "save_pareto_plot", lambda *args: Path("out") / "figures" / "pareto.png")
    monkeypatch.setattr(run_optimization, "analyze_pymoo_convergence", lambda *args: {"summary": {}, "metrics": pd.DataFrame()})
    monkeypatch.setattr(run_optimization, "save_convergence_plots", lambda *args: {})
    monkeypatch.setattr(run_optimization, "save_pareto_distribution_plot", lambda *args: None)
    monkeypatch.setattr(run_optimization, "_resolve_output_dir", lambda loaded_config, config_file: Path("out"))
    monkeypatch.setattr(run_optimization, "write_report", lambda *args: Path("out") / "report.md")
    monkeypatch.setattr(run_optimization, "_write_pareto_artifacts", lambda *args: [])
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

    def fake_spatial_analysis(**kwargs):
        spatial_calls.append(kwargs)
        return {"available": True, "solution_id": 0}

    monkeypatch.setattr(run_optimization, "run_spatial_analysis", fake_spatial_analysis)

    output = run_optimization.run("config.json")

    assert output["spatial_result"] == {"available": True, "solution_id": 0}
    assert len(spatial_calls) == 1
    assert spatial_calls[0]["rack_metadata"].equals(data.rack_metadata)
    assert spatial_calls[0]["hourly"].equals(data.hourly)
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run:

```powershell
python -m pytest tests\test_reporting.py::test_run_calls_spatial_analysis_for_non_empty_pareto -q -p no:cacheprovider
```

Expected: failure because `run_optimization` does not import or call `run_spatial_analysis`.

- [ ] **Step 3: Import and call spatial analysis**

In `run_optimization.py`, add the import:

```python
from spatial_analysis import run_spatial_analysis
```

After:

```python
distribution_path = save_pareto_distribution_plot(pareto, output_dir, config) if not pareto.empty else None
```

add:

```python
spatial_result = run_spatial_analysis(
    pareto=pareto,
    rack_metadata=getattr(data, "rack_metadata", pd.DataFrame()),
    hourly=getattr(data, "hourly", pd.DataFrame()),
    config=config,
    output_dir=output_dir,
) if not pareto.empty else {"available": False, "reason": "empty_pareto"}
```

In the returned dictionary, add:

```python
"spatial_result": spatial_result,
```

- [ ] **Step 4: Run the integration test**

Run:

```powershell
python -m pytest tests\test_reporting.py::test_run_calls_spatial_analysis_for_non_empty_pareto -q -p no:cacheprovider
```

Expected: pass.

- [ ] **Step 5: Run the empty-Pareto regression test**

Run:

```powershell
python -m pytest tests\test_reporting.py::test_run_exports_empty_pareto_without_plot_when_all_solutions_infeasible -q -p no:cacheprovider
```

Expected: pass. The test may need one extra assertion:

```python
assert result["spatial_result"]["available"] is False
```

---

### Task 5: Add Spatial Section to the Markdown Report

**Files:**
- Modify: `D:\paper\cooling_retrofit_code\analyze_results.py`
- Modify: `D:\paper\cooling_retrofit_code\tests\test_reporting.py`

- [ ] **Step 1: Write a failing report test**

Add this test to `tests\test_reporting.py`:

```python
def test_write_report_includes_spatial_artifacts_when_present(monkeypatch, tmp_path):
    output = tmp_path
    figures = output / "figures"
    spatial = output / "spatial"
    figures.mkdir()
    spatial.mkdir()
    (figures / "room_layout_knee_solution.png").write_text("png", encoding="utf-8")
    (figures / "config_choice_vs_growth_age.png").write_text("png", encoding="utf-8")
    pd.DataFrame(
        {
            "ac_unit": ["cdz1", "cdz2"],
            "config_name": ["existing_ac", "rdhx_wshp"],
            "rho_load_mean": [0.08, 0.20],
            "alpha_age": [0.05, 0.25],
            "zone_peak_it_kw": [50.0, 80.0],
        }
    ).to_csv(spatial / "config_choice_relationship.csv", index=False, encoding="utf-8-sig")

    written = {}
    original_write_text = Path.write_text

    def capture_write(path, text, encoding=None):
        if path.name == "report.md":
            written["text"] = text
            return len(text)
        return original_write_text(path, text, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", capture_write)
    monkeypatch.setattr(Path, "exists", lambda path: path.name in {
        "room_layout_knee_solution.png",
        "config_choice_vs_growth_age.png",
        "config_choice_relationship.csv",
    })

    all_evaluations = pd.DataFrame({"generation": [0], "feasible": [True], "tlcc": [100.0], "tce": [80.0]})
    pareto = all_evaluations.copy()

    write_report(all_evaluations, pareto, {"assumptions": {}}, output)

    report = written["text"]
    assert "## 7. 机房空间与构型解释" in report
    assert "room_layout_knee_solution.png" in report
    assert "config_choice_vs_growth_age.png" in report
    assert "cdz2" in report
    assert "rdhx_wshp" in report
```

- [ ] **Step 2: Run the report test to verify it fails**

Run:

```powershell
python -m pytest tests\test_reporting.py::test_write_report_includes_spatial_artifacts_when_present -q -p no:cacheprovider
```

Expected: failure because the section is not written.

- [ ] **Step 3: Add report helpers**

In `analyze_results.py`, add:

```python
def _spatial_lines(output: Path) -> list[str]:
    figures = output / "figures"
    spatial = output / "spatial"
    room_plot = figures / "room_layout_knee_solution.png"
    relationship_plot = figures / "config_choice_vs_growth_age.png"
    relationship_csv = spatial / "config_choice_relationship.csv"
    lines = ["## 7. 机房空间与构型解释", ""]
    if not room_plot.exists() or not relationship_plot.exists() or not relationship_csv.exists():
        lines.append("- 本轮未生成完整空间可视化产物；若 Pareto 为空或缺少机柜坐标/业务爬升/老化参数，将跳过该部分。")
        return lines
    lines.extend(
        [
            f"![机房空间构型图]({room_plot})",
            "",
            f"![业务爬升与空调老化关系图]({relationship_plot})",
            "",
            "- 空间图按机柜 `row/col` 坐标展示 Pareto 折中解的构型选择；点内编号为机柜编号，颜色为改造构型。",
            "- 关系图以空调组为样本点，横轴为区域平均业务爬升强度，纵轴为旧空调容量老化率；该图用于解释优化结果的描述性关联，不作为因果检验。",
        ]
    )
    try:
        relation = pd.read_csv(relationship_csv)
    except (OSError, pd.errors.EmptyDataError):
        return lines
    if relation.empty:
        return lines
    ordered = relation.copy()
    if "alpha_age" in ordered.columns:
        ordered["alpha_age_num"] = pd.to_numeric(ordered["alpha_age"], errors="coerce")
        ordered = ordered.sort_values("alpha_age_num", ascending=False)
    lines.append("- 老化率较高空调组的构型选择摘要：")
    for _, row in ordered.head(3).iterrows():
        lines.append(
            f"  - `{row.get('ac_unit', 'N/A')}`：config={row.get('config_name', 'N/A')}，"
            f"rho_mean={_format_percent(_to_float(row.get('rho_load_mean'), float('nan')))}，"
            f"alpha_age={_format_percent(_to_float(row.get('alpha_age'), float('nan')))}，"
            f"peak_IT={_format_number(_to_float(row.get('zone_peak_it_kw'), float('nan')), 'kW')}"
        )
    return lines
```

- [ ] **Step 4: Insert the spatial section before the output index**

In `write_report(...)`, after the feasibility/repair section and before the `figures = output / "figures"` block, insert:

```python
lines.extend(["", *_spatial_lines(output)])
```

Then renumber the later section headings:

- `## 7. 输出文件索引` becomes `## 8. 输出文件索引`
- `## 8. 参数来源与假设` becomes `## 9. 参数来源与假设`
- `## 9. 后续审核重点` becomes `## 10. 后续审核重点`

In the output index list, add:

```python
f"- 机房空间构型图：`{figures / 'room_layout_knee_solution.png'}`",
f"- 业务爬升-空调老化关系图：`{figures / 'config_choice_vs_growth_age.png'}`",
f"- 机柜空间构型明细：`{output / 'spatial' / 'knee_solution_rack_layout.csv'}`",
f"- 构型选择关系明细：`{output / 'spatial' / 'config_choice_relationship.csv'}`",
```

- [ ] **Step 5: Run report tests**

Run:

```powershell
python -m pytest tests\test_reporting.py -q -p no:cacheprovider
```

Expected: all reporting tests pass.

---

### Task 6: Run a Focused Verification Set

**Files:**
- Reads all touched files.
- Writes only normal result artifacts if `run_optimization.py` is executed manually.

- [ ] **Step 1: Run focused unit tests**

Run:

```powershell
python -m pytest tests\test_spatial_analysis.py tests\test_reporting.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 2: Run existing convergence/reporting tests**

Run:

```powershell
python -m pytest tests\test_convergence_analysis.py tests\test_reporting.py tests\test_spatial_analysis.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run the project once only if runtime is acceptable**

Run:

```powershell
python run_optimization.py config.json
```

Expected:

- `results\figures\room_layout_knee_solution.png` exists.
- `results\figures\config_choice_vs_growth_age.png` exists.
- `results\spatial\knee_solution_rack_layout.csv` exists.
- `results\spatial\config_choice_relationship.csv` exists.
- `results\report.md` includes the spatial section.

Skip this step if the configured NSGA-II run is too long for the current session; the unit tests are the required verification baseline.

---

## Self-Review

- Spec coverage:
  - Static room layout: Task 2 implements and Task 5 reports it.
  - Growth-aging relationship using `alpha_z_age`: Task 2 implements and Task 3 configures it.
  - Knee solution only: Task 2 `select_knee_solution`.
  - CSV outputs: Task 2 writes both CSV files.
  - Report integration: Task 5.
  - Minimal persistent tests: Task 1 and Task 5 use temporary directories and disabled pytest cache.
- Placeholder scan:
  - The plan contains no incomplete placeholder sections.
  - The only skipped action is the full optimization run when runtime is not acceptable; focused tests remain mandatory.
- Type consistency:
  - The public module functions used in tests match the implementation task names.
  - `rho_load` maps rack-level values first, then AC-unit values, then default values.
  - `alpha_age` maps AC-unit values, then default values.
