import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

import spatial_analysis
from spatial_analysis import (
    build_rack_layout_frame,
    build_zone_relationship_frame,
    config_name_map,
    run_spatial_analysis,
    save_room_layout_plot,
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


def test_config_name_map_casts_ids_to_ints():
    assert config_name_map(_config()) == {
        0: "existing_ac",
        4: "cold_plate_cdu_wshp",
        5: "rdhx_wshp",
    }


def test_build_rack_layout_frame_maps_config_growth_and_age():
    row = select_knee_solution(_pareto())

    frame = build_rack_layout_frame(row, _rack_metadata(), _config())

    assert frame["solution_id"].tolist() == [3, 3, 3]
    assert frame["config_id"].tolist() == [4, 4, 5]
    assert frame["config_name"].tolist() == ["cold_plate_cdu_wshp", "cold_plate_cdu_wshp", "rdhx_wshp"]
    assert frame["rho_load"].tolist() == [0.12, 0.12, 0.20]
    assert frame["alpha_age"].tolist() == [0.25, 0.25, 0.05]
    assert frame["epsilon_age"].tolist() == [0.25, 0.25, 0.05]


def test_build_rack_layout_frame_prefers_simulated_metadata_values():
    row = select_knee_solution(_pareto())
    metadata = _rack_metadata().assign(
        rho_load=[0.11, 0.13, 0.21],
        alpha_age=[0.22, 0.22, 0.09],
        epsilon_age=[0.22, 0.22, 0.09],
    )

    frame = build_rack_layout_frame(row, metadata, _config())

    assert frame["rho_load"].tolist() == [0.11, 0.13, 0.21]
    assert frame["alpha_age"].tolist() == [0.22, 0.22, 0.09]


def test_build_rack_layout_frame_keeps_replicated_room_ids():
    row = pd.Series(
        {
            "solution_id": 7,
            "config_by_zone_json": json.dumps(
                {
                    "room01_cdz1": 4,
                    "room02_cdz1": 5,
                }
            ),
        }
    )
    metadata = pd.DataFrame(
        {
            "rack_id": ["room01_rack_it_000", "room02_rack_it_000"],
            "row": [2, 2],
            "col": [21, 21],
            "ac_unit": ["room01_cdz1", "room02_cdz1"],
            "room_id": ["room01", "room02"],
            "rho_load": [0.10, 0.22],
            "alpha_age": [0.08, 0.18],
        }
    )

    frame = build_rack_layout_frame(row, metadata, _config())

    assert frame["room_id"].tolist() == ["room01", "room02"]
    assert frame["config_id"].tolist() == [4, 5]
    assert frame["plot_col"].tolist() == [21.0, 21.0]
    assert frame["plot_row"].tolist() == [2.0, 2.0]


def test_save_room_layout_plot_writes_one_image_per_room():
    row = pd.Series(
        {
            "solution_id": 7,
            "tlcc": 100.0,
            "tce": 20.0,
            "config_by_zone_json": json.dumps(
                {
                    "room01_cdz1": 4,
                    "room02_cdz1": 5,
                }
            ),
        }
    )
    metadata = pd.DataFrame(
        {
            "rack_id": ["room01_rack_it_000", "room02_rack_it_000"],
            "row": [2, 2],
            "col": [21, 21],
            "ac_unit": ["room01_cdz1", "room02_cdz1"],
            "room_id": ["room01", "room02"],
            "rho_load": [0.10, 0.22],
            "alpha_age": [0.08, 0.18],
        }
    )
    frame = build_rack_layout_frame(row, metadata, _config())

    output_dir = (
        Path(__file__).resolve().parents[1]
        / ".test_tmp"
        / f"spatial_one_image_per_room_{uuid.uuid4().hex}"
    )
    shutil.rmtree(output_dir, ignore_errors=True)

    path = save_room_layout_plot(frame, row, output_dir)

    assert path.name == "room_layout_knee_solution_room01.png"
    assert (output_dir / "figures" / "room_layout_knee_solution_room01.png").exists()
    assert (output_dir / "figures" / "room_layout_knee_solution_room02.png").exists()
    assert not (output_dir / "figures" / "room_layout_knee_solution.png").exists()


def test_build_zone_relationship_frame_aggregates_peak_load_and_hotspot():
    row = select_knee_solution(_pareto())
    rack_frame = build_rack_layout_frame(row, _rack_metadata(), _config())

    frame = build_zone_relationship_frame(row, rack_frame, _hourly(), _config())

    cdz1 = frame.set_index("ac_unit").loc["cdz1"]
    cdz2 = frame.set_index("ac_unit").loc["cdz2"]
    assert float(cdz1["rho_load_mean"]) == 0.12
    assert float(cdz1["alpha_age"]) == 0.25
    assert float(cdz1["zone_peak_it_kw"]) == 40.0
    assert float(cdz1["zone_max_hotspot_risk"]) == 0.4
    assert float(cdz2["rho_load_mean"]) == 0.20
    assert float(cdz2["zone_peak_it_kw"]) == 50.0


def test_run_spatial_analysis_writes_expected_artifacts(monkeypatch):
    written_csv = []

    def _fake_room_plot(rack_frame, knee_row, output_dir):
        del rack_frame, knee_row
        return Path(output_dir) / "figures" / "room_layout_knee_solution.png"

    def _fake_relationship_plot(zone_frame, output_dir):
        del zone_frame
        return Path(output_dir) / "figures" / "config_choice_vs_growth_age.png"

    def _fake_to_csv(self, path, *args, **kwargs):
        del self, args, kwargs
        written_csv.append(Path(path).name)

    monkeypatch.setattr(Path, "mkdir", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", _fake_to_csv)
    monkeypatch.setattr(spatial_analysis, "save_room_layout_plot", _fake_room_plot)
    monkeypatch.setattr(spatial_analysis, "save_growth_age_plot", _fake_relationship_plot)

    result = run_spatial_analysis(
        pareto=_pareto(),
        rack_metadata=_rack_metadata(),
        hourly=_hourly(),
        config=_config(),
        output_dir=Path("virtual_spatial_output"),
    )

    assert result["available"] is True
    assert Path(result["room_layout_path"]).name == "room_layout_knee_solution.png"
    assert Path(result["relationship_path"]).name == "config_choice_vs_growth_age.png"
    assert sorted(written_csv) == ["config_choice_relationship.csv", "knee_solution_rack_layout.csv"]
