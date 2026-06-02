import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gurobipy")

from data_utils import load_config, load_input_data
from decision import DecisionSchema
from feasibility_oracle import FeasibilityOracle
from inner_dispatch import InnerDispatchMILP, InnerSolveResult, _rack_zone_map
from model import _capacity_cost, _crf, _floor_weight_margin_by_zone, evaluate_solution
from repair import repair_vector, screen_decision
from typical_days import build_time_index


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((PROJECT_DIR / "config.json").read_text(encoding="utf-8"))


def _time_frame(load_kw=100.0, heat_kw=0.0, periods=24) -> pd.DataFrame:
    per_rack = load_kw / 2.0
    return pd.DataFrame(
        {
            "timestamp_hour_utc": pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC"),
            "rack_it_000_kw": [per_rack] * periods,
            "rack_it_001_kw": [per_rack] * periods,
            "it_load_kw": [load_kw] * periods,
            "price_yuan_per_kwh": [0.5] * periods,
            "carbon_kg_per_kwh": [0.5] * periods,
            "outdoor_temp_c": [20.0] * periods,
            "heating_demand_kw": [heat_kw] * periods,
            "representative_day_id": [0] * periods,
            "day_weight": [1.0] * periods,
        }
    )


def _minimal_config() -> dict:
    return {
        "economics": {
            "discount_rate": 0.0,
            "project_life_years": 10,
            "heat_credit_yuan_per_kwh": 0.0,
            "capex_yuan_per_kw": {},
            "retrofit_fixed_yuan": {},
        },
        "carbon": {
            "allow_negative_operational_carbon": False,
            "heat_displacement_kg_per_kwh": 0.0,
            "embodied_kg_per_kw": {},
            "retrofit_fixed_kg": {},
        },
        "scenario": {},
        "technology": {},
        "solver": {},
    }


def test_decision_schema_preserves_all_new_model_options_and_capacity_layout():
    schema = DecisionSchema(["z1", "z2"])
    lower, upper = schema.bounds()
    vector = schema.encode_default()
    vector[0] = 2
    vector[1] = 5
    vector[2:] = np.arange(1, schema.vector_length - 1, dtype=float)

    decision = schema.decode(vector)

    assert len(lower) == len(upper) == schema.vector_length
    assert decision.s_z == {"z1": 2, "z2": 5}
    assert decision.n_z["z1"] == 1
    assert decision.a_z["z1"] == 1
    assert decision.r_z["z2"] == 1
    assert decision.cap_ac_new == {"z1": 1.0, "z2": 2.0}
    assert not hasattr(decision, "cap_vent")
    assert decision.cap_rdhx == {"z1": 3.0, "z2": 4.0}
    assert decision.cap_cdu == {"z1": 5.0, "z2": 6.0}
    assert decision.cap_ashp == 7.0
    assert decision.cap_wshp == 8.0
    assert decision.cap_bess == 9.0
    assert not hasattr(decision, "cap_bess_e")
    assert not hasattr(decision, "cap_bess_p")
    assert decision.cap_tes == 10.0
    assert decision.delta_cap_chiller == 11.0
    assert decision.delta_cap_tower == 12.0
    assert upper[-6] > 0.0


def test_decision_schema_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Duplicate zone_ids"):
        DecisionSchema(["z1", "z1"])

    schema = DecisionSchema(["z1"])
    with pytest.raises(ValueError, match="Expected vector length"):
        schema.decode(np.zeros(schema.vector_length - 1))
    vector = schema.encode_default()
    vector[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        schema.decode(vector)


def test_decision_schema_uses_configured_capacity_bounds():
    schema = DecisionSchema.from_config(
        ["z1", "z2"],
        {
            "decision_bounds": {
                "default_zone_cap_upper_kw": 7.0,
                "default_system_cap_upper_kw": 8.0,
                "zone_cap_upper_kw": {
                    "cap_ac_new": 11.0,
                    "cap_cdu": 44.0,
                },
                "system_cap_upper_kw": {
                    "cap_bess": 99.0,
                    "delta_cap_tower": 123.0,
                },
            }
        },
    )
    _lower, upper = schema.bounds()

    zone_count = len(schema.zone_ids)
    start = zone_count
    offsets = {
        name: start + idx * zone_count
        for idx, name in enumerate(schema.zone_capacity_names)
    }
    system_start = zone_count + len(schema.zone_capacity_names) * zone_count
    system_offsets = {
        name: system_start + idx
        for idx, name in enumerate(schema.system_capacity_names)
    }

    assert upper[offsets["cap_ac_new"]] == pytest.approx(11.0)
    assert "cap_vent" not in offsets
    assert upper[offsets["cap_rdhx"]] == pytest.approx(7.0)
    assert upper[offsets["cap_cdu"]] == pytest.approx(44.0)
    assert upper[system_offsets["cap_ashp"]] == pytest.approx(8.0)
    assert upper[system_offsets["cap_bess"]] == pytest.approx(99.0)
    assert upper[system_offsets["delta_cap_tower"]] == pytest.approx(123.0)


def test_replicated_room_rack_zone_map_uses_room_prefixed_original_metadata():
    config = load_config(PROJECT_DIR / "config.json")
    rack_ids = ["room01_rack_it_000", "room02_rack_it_000"]
    zone_ids = ["room01_cdz5", "room02_cdz5"]

    assert _rack_zone_map(config, rack_ids, zone_ids) == zone_ids


def test_feasibility_oracle_repairs_screens_and_remembers_infeasible_signatures():
    schema = DecisionSchema(["z1"], zone_cap_upper_kw=50.0)
    oracle = FeasibilityOracle(
        schema=schema,
        peak_load_by_zone_kw={"z1": 100.0},
        floor_weight_margin_by_zone_kg={"z1": 2000.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )
    vector = schema.encode_default()
    vector[0] = 5

    first = oracle.evaluate(vector)
    second = oracle.evaluate(vector)

    assert first.feasible is False
    assert "peak_shortage" in "|".join(first.screening.reasons)
    assert second.cached_infeasible is True
    assert second.screening.feasible is False


def test_repair_clears_incompatible_capacity_without_hard_weight_scaling():
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 2
    vector[1] = 100.0
    vector[2] = 100.0
    vector[3] = 100.0

    repaired, actions = repair_vector(vector, schema, {"z1": 80.0}, {"z1": 120.0})
    decision = schema.decode(repaired)

    assert decision.cap_ac_new["z1"] == pytest.approx(100.0)
    assert decision.cap_cdu["z1"] == 0.0
    assert decision.cap_rdhx["z1"] == 0.0
    assert decision.cap_ashp >= 0.0
    assert any("clear_rdhx" in action for action in actions)
    assert not any("weight_margin" in action for action in actions)


def test_screening_records_weight_excess_as_soft_violation_without_blocking():
    schema = DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 1
    vector[1] = 100.0
    decision = schema.decode(vector)

    screening = screen_decision(
        decision,
        {"z1": 100.0},
        {"z1": 100.0},
        chiller_old_kw=200.0,
        tower_old_kw=300.0,
        equipment_kg_per_kw={"ac_new": 5.0},
    )

    assert screening.feasible is True
    assert screening.violation == pytest.approx(0.0)
    assert screening.soft_violation == pytest.approx(400.0)
    assert "weight_margin_exceeded" in "|".join(screening.soft_reasons)


def test_screening_blocks_clear_shortages_but_allows_non_chiller_and_recovered_heat_paths():
    schema = DecisionSchema(["z1"])

    shortage = schema.encode_default()
    shortage[0] = 4
    repaired, _ = repair_vector(shortage, schema, {"z1": 100.0}, {"z1": 2000.0})
    assert not screen_decision(
        schema.decode(repaired),
        {"z1": 500.0},
        {"z1": 2000.0},
        chiller_old_kw=0.0,
    ).feasible

    cdu = schema.encode_default()
    cdu[0] = 4
    cdu[3] = 100.0
    cdu[5] = 200.0
    cdu[-2] = 30.0
    cdu[-1] = 40.0
    assert screen_decision(
        schema.decode(cdu),
        {"z1": 100.0},
        {"z1": 2000.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
        peak_heating_demand_kw=200.0,
        cop_wshp=4.0,
    ).feasible

    rdhx = schema.encode_default()
    rdhx[0] = 5
    rdhx[2] = 100.0
    rdhx[5] = 200.0
    assert screen_decision(
        schema.decode(rdhx),
        {"z1": 100.0},
        {"z1": 2000.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
        peak_heating_demand_kw=200.0,
        cop_wshp=4.0,
    ).feasible


def test_dispatch_keeps_cdu_free_cooling_out_of_chiller_evaporator_load():
    schema = DecisionSchema(["z1"])
    config = _config()
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["cold_plate"] = 1.0
    vector = schema.encode_default()
    vector[0] = 4
    vector[3] = 100.0
    vector[-1] = 150.0
    decision = schema.decode(vector)

    result = InnerDispatchMILP(config).solve(
        decision=decision,
        time_frame=_time_frame(),
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )

    assert result.feasible is True
    assert result.dispatch["q_cp_to_free_kw"].sum() == pytest.approx(2400.0)
    assert result.dispatch["q_chiller_evap_kw"].sum() == pytest.approx(0.0)
    for column in [
        "e_tes_kwh",
        "q_tes_ch_kw",
        "q_tes_dis_kw",
        "e_bess_kwh",
        "p_bess_ch_kw",
        "p_bess_dis_kw",
        "s_air_kw",
        "max_theta_c",
        "hotspot_slack_c",
        "power_balance_residual_kw",
        "heat_balance_residual_kw",
    ]:
        assert column in result.dispatch.columns
    assert "q_heat_unserved_kw" not in result.dispatch.columns
    assert "heat_unserved_penalty_yuan" not in result.dispatch.columns
    assert result.diagnostics["max_power_balance_residual_kw"] <= 1e-5
    assert result.diagnostics["max_heat_balance_residual_kw"] <= 1e-5


def test_dispatch_treats_heating_demand_as_hard_constraint_without_unserved_penalty():
    schema = DecisionSchema(["z1"])
    config = _config()
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["cold_plate"] = 1.0
    vector = schema.encode_default()
    vector[0] = 4
    vector[3] = 100.0
    vector[-1] = 150.0

    result = InnerDispatchMILP(config).solve(
        decision=schema.decode(vector),
        time_frame=_time_frame(load_kw=100.0, heat_kw=50.0),
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )

    assert result.feasible is False
    assert result.dispatch.empty
    assert result.diagnostics["reason"] == "no_solution"


def test_dispatch_air_service_tracks_air_load_and_nested_solver_config():
    schema = DecisionSchema(["z1"])
    config = _config()
    config["solver"] = {"gurobi": {"time_limit_seconds": 5, "mip_gap": 0.05, "threads": 1}}
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["cold_plate"] = 0.0
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["rdhx"] = 0.0
    vector = schema.encode_default()
    vector[0] = 1
    vector[1] = 100.0
    vector[-2] = 100.0
    vector[-1] = 150.0

    result = InnerDispatchMILP(config).solve(
        decision=schema.decode(vector),
        time_frame=_time_frame(),
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )

    assert result.feasible is True
    assert result.dispatch["s_air_kw"].tolist() == pytest.approx(result.dispatch["q_air_kw"].tolist())
    assert result.diagnostics["gurobi_time_limit_seconds"] == pytest.approx(5.0)
    assert result.diagnostics["gurobi_mip_gap"] == pytest.approx(0.05)


def test_dispatch_uses_aged_old_ac_capacity_and_terminal_coefficient():
    schema = DecisionSchema(["z1"])
    config = _config()
    config["solver"] = {"gurobi": {"time_limit_seconds": 5, "mip_gap": 0.05, "threads": 1}}
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["cold_plate"] = 0.0
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["rdhx"] = 0.0
    frame = _time_frame(load_kw=80.0)
    frame["old_ac_capacity_eff_z1_kw"] = 80.0
    frame["old_ac_terminal_coeff_z1_kw_per_kw"] = 1.0
    frame["psi_amb_z1"] = 1.0
    vector = schema.encode_default()
    vector[0] = 0

    result = InnerDispatchMILP(config).solve(
        decision=schema.decode(vector),
        time_frame=frame,
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=100.0,
        tower_old_kw=150.0,
    )

    assert result.feasible is True
    assert result.dispatch["s_air_old_kw"].tolist() == pytest.approx([80.0] * 24)
    assert result.dispatch["s_air_new_kw"].tolist() == pytest.approx([0.0] * 24)
    assert result.dispatch["old_ac_capacity_eff_kw"].tolist() == pytest.approx([80.0] * 24)
    assert result.dispatch["p_terminal_kw"].tolist() == pytest.approx([80.0] * 24)


def test_dispatch_uses_rdhx_or_cdu_as_wshp_source_when_heat_is_demanded():
    schema = DecisionSchema(["z1"])
    config = _config()
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["cold_plate"] = 1.0
    config.setdefault("technology", {}).setdefault("liquid_heat_fraction", {})["rdhx"] = 1.0

    rdhx = schema.encode_default()
    rdhx[0] = 5
    rdhx[2] = 100.0
    rdhx[5] = 100.0
    rdhx[-2] = 100.0
    rdhx[-1] = 150.0
    rdhx_result = InnerDispatchMILP(config).solve(
        decision=schema.decode(rdhx),
        time_frame=_time_frame(heat_kw=50.0),
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )

    cdu = schema.encode_default()
    cdu[0] = 4
    cdu[3] = 100.0
    cdu[5] = 100.0
    cdu[-1] = 150.0
    cdu_result = InnerDispatchMILP(config).solve(
        decision=schema.decode(cdu),
        time_frame=_time_frame(heat_kw=50.0),
        peak_load_by_zone_kw={"z1": 100.0},
        chiller_old_kw=0.0,
        tower_old_kw=0.0,
    )

    source_fraction = 1.0 - 1.0 / config["technology"]["cop_wshp"]
    assert rdhx_result.feasible is True
    assert rdhx_result.dispatch["q_rdhx_to_wshp_kw"].sum() == pytest.approx(
        (rdhx_result.dispatch["q_heat_kw"] * source_fraction).sum()
    )
    assert cdu_result.feasible is True
    assert cdu_result.dispatch["q_cp_to_wshp_kw"].sum() == pytest.approx(
        (cdu_result.dispatch["q_heat_kw"] * source_fraction).sum()
    )


def test_evaluate_solution_with_real_typical_days_returns_feasible_dispatch():
    config = load_config(PROJECT_DIR / "config.json")
    config.setdefault("weight", {})["floor_weight_margin_by_zone_kg"] = {"zone_1": 1.0e9}
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])

    schema = DecisionSchema(["zone_1"], zone_cap_upper_kw=100000.0, system_cap_upper_kw=100000.0)
    peak = float(time_index.frame["it_load_kw"].max())
    heat_peak = float(time_index.frame["heating_demand_kw"].max())
    vector = schema.encode_default()
    vector[0] = 4
    vector[3] = peak
    vector[4] = heat_peak
    vector[5] = heat_peak
    vector[7] = heat_peak * 8.0
    vector[-1] = peak * 1.25

    result = evaluate_solution(vector, schema, config, time_index.frame, {"zone_1": peak})

    assert result.feasible is True
    assert result.tlcc >= 0.0
    assert result.tce >= 0.0
    assert not result.artifacts["dispatch"].empty


def test_evaluate_solution_short_circuits_screening_before_dispatch(monkeypatch):
    schema = DecisionSchema(["zone_1"], zone_cap_upper_kw=50.0)
    vector = schema.encode_default()
    vector[0] = 5

    class SolverThatShouldNotRun:
        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            raise AssertionError("dispatch should not run after failed screening")

    monkeypatch.setattr("model.InnerDispatchMILP", SolverThatShouldNotRun)

    result = evaluate_solution(vector, schema, _minimal_config(), _time_frame(10.0), {"zone_1": 100.0})

    assert result.feasible is False
    assert result.tlcc == float("inf")
    assert "dispatch" not in result.artifacts
    assert "repair_actions" in result.artifacts


def test_floor_weight_margins_support_explicit_and_uniform_area_inputs():
    explicit = _minimal_config()
    explicit["weight"] = {
        "floor_limit_kg_per_m2": 800.0,
        "zone_floor_area_m2_by_zone": {"zone_1": 2.0},
        "rack_static_kg": 900.0,
    }
    uniform = _minimal_config()
    uniform["weight"] = {
        "floor_limit_kg_per_m2": 800.0,
        "zone_floor_area_m2": 12.0,
        "rack_static_kg": 900.0,
    }

    assert _floor_weight_margin_by_zone(explicit, ["zone_1"]) == {"zone_1": 700.0}
    assert _floor_weight_margin_by_zone(uniform, ["zone_1", "zone_2"]) == {
        "zone_1": 8700.0,
        "zone_2": 8700.0,
    }


def test_operating_and_capital_objectives_use_row_weights_and_fixed_retrofit_costs(monkeypatch):
    schema = DecisionSchema(["zone_1"])
    vector = schema.encode_default()
    vector[0] = 4
    vector[3] = 10.0
    dispatch = pd.DataFrame({"p_grid_kw": [10.0, 10.0], "q_heat_kw": [0.0, 0.0], "day_weight": [1.0, 3.0]})

    class FakeSolver:
        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            return InnerSolveResult(True, 0.0, dispatch, {"fake": True})

    monkeypatch.setattr("model.InnerDispatchMILP", FakeSolver)
    time_frame = pd.DataFrame(
        {
            "timestamp_hour_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            "it_load_kw": [10.0, 10.0],
            "price_yuan_per_kwh": [1.0, 3.0],
            "carbon_kg_per_kwh": [0.5, 0.5],
            "heating_demand_kw": [0.0, 0.0],
            "day_weight": [1.0, 3.0],
        }
    )
    result = evaluate_solution(vector, schema, _minimal_config(), time_frame, {"zone_1": 10.0})
    assert result.tlcc == pytest.approx(100.0)

    decision = schema.decode(vector)
    config = _minimal_config()
    config["economics"]["capex_yuan_per_kw"] = {"cdu": 50.0}
    config["economics"]["retrofit_fixed_yuan"] = {"4": 1000.0}
    config["carbon"]["embodied_kg_per_kw"] = {"cdu": 5.0}
    config["carbon"]["retrofit_fixed_kg"] = {"4": 100.0}
    annualized_cost, annualized_carbon = _capacity_cost(decision, config)

    assert _crf(0.0, 20) == pytest.approx(0.05)
    assert annualized_cost == pytest.approx(150.0)
    assert annualized_carbon == pytest.approx(15.0)


def test_evaluate_solution_adds_weight_soft_penalty_to_objectives(monkeypatch):
    schema = DecisionSchema(["zone_1"])
    vector = schema.encode_default()
    vector[0] = 1
    vector[1] = 100.0
    dispatch = pd.DataFrame({"p_grid_kw": [0.0], "q_heat_kw": [0.0], "day_weight": [1.0]})

    class FakeSolver:
        def __init__(self, config):
            pass

        def solve(self, *args, **kwargs):
            return InnerSolveResult(True, 0.0, dispatch, {"fake": True})

    monkeypatch.setattr("model.InnerDispatchMILP", FakeSolver)
    config = _minimal_config()
    config["technology"] = {"existing_chiller_kw": 1000.0, "existing_tower_kw": 1000.0}
    config["weight"] = {
        "floor_weight_margin_by_zone_kg": {"zone_1": 100.0},
        "equipment_kg_per_kw": {"ac_new": 5.0},
        "soft_constraint": {
            "enabled": True,
            "cost_penalty_yuan_per_kg_year": 2.0,
            "carbon_penalty_kg_per_kg_year": 0.5,
        },
    }

    result = evaluate_solution(
        vector,
        schema,
        config,
        pd.DataFrame(
            {
                "timestamp_hour_utc": pd.date_range("2025-01-01", periods=1, freq="h", tz="UTC"),
                "it_load_kw": [100.0],
                "price_yuan_per_kwh": [0.0],
                "carbon_kg_per_kwh": [0.0],
                "heating_demand_kw": [0.0],
                "day_weight": [1.0],
            }
        ),
        {"zone_1": 100.0},
    )

    assert result.feasible is True
    assert result.tlcc == pytest.approx(800.0)
    assert result.tce == pytest.approx(200.0)
    assert result.artifacts["screening"].soft_violation == pytest.approx(400.0)
    assert result.artifacts["weight_soft_penalty"]["cost_yuan_per_year"] == pytest.approx(800.0)
