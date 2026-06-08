import json

import numpy as np
import pandas as pd
import pytest

from benchmark_scenarios import _add_payback_metrics, evaluate_benchmark_scenarios
from decision import DecisionSchema
from model import EvaluationResult


def test_benchmark_scenarios_build_no_retrofit_aggressive_and_knee_rows():
    schema = DecisionSchema(["z1", "z2"], zone_cap_upper_kw=1000.0, system_cap_upper_kw=5000.0)
    time_frame = pd.DataFrame(
        {
            "it_load_kw": [300.0, 500.0],
            "heating_demand_kw": [120.0, 220.0],
            "old_ac_capacity_eff_z1_kw": [100.0, 100.0],
            "old_ac_capacity_eff_z2_kw": [250.0, 250.0],
            "day_weight": [2.0, 3.0],
            "price_yuan_per_kwh": [1.0, 2.0],
            "carbon_kg_per_kwh": [0.5, 0.6],
        }
    )
    peak_by_zone = {"z1": 150.0, "z2": 350.0}
    pareto_vectors = []
    for k1, k2 in ((1, 1), (4, 4), (2, 4)):
        vector = schema.encode_default()
        vector[0] = k1
        vector[1] = k2
        pareto_vectors.append(json.dumps(vector.tolist()))
    pareto = pd.DataFrame(
        {
            "solution_id": [11, 12, 13],
            "tlcc": [900.0, 500.0, 650.0],
            "tce": [200.0, 500.0, 260.0],
            "raw_vector_json": pareto_vectors,
            "config_by_zone_json": [
                json.dumps({"z1": 1, "z2": 1}),
                json.dumps({"z1": 4, "z2": 4}),
                json.dumps({"z1": 4, "z2": 6}),
            ],
        }
    )
    decoded_vectors = []

    def fake_evaluate(vector, schema_arg, config, frame, peaks):
        assert schema_arg is schema
        assert frame is time_frame
        assert peaks is peak_by_zone
        decoded_vectors.append(schema.decode(vector))
        config_id_sum = sum(decoded_vectors[-1].s_z.values())
        return EvaluationResult(
            feasible=True,
            tlcc=1000.0 + config_id_sum,
            tce=2000.0 + config_id_sum,
            artifacts={
                "repaired_vector": np.asarray(vector, dtype=float),
                "decision": decoded_vectors[-1],
                "dispatch": pd.DataFrame(
                    {
                        "it_load_kw": [100.0, 300.0],
                        "q_heat_kw": [20.0, 80.0],
                        "p_cooling_kw": [20.0, 60.0],
                        "p_grid_kw": [120.0, 360.0],
                        "day_weight": [2.0, 3.0],
                    }
                ),
            },
        )

    comparison = evaluate_benchmark_scenarios(
        schema=schema,
        config={
            "benchmarks": {"aggressive_config_id": 6},
            "economics": {"heat_credit_yuan_per_kwh": 0.1},
            "carbon": {"heat_displacement_kg_per_kwh": 0.2},
        },
        time_frame=time_frame,
        peak_load_by_zone_kw=peak_by_zone,
        pareto=pareto,
        evaluate_fn=fake_evaluate,
    )

    assert comparison["scenario_key"].tolist() == [
        "baseline_1_no_retrofit",
        "baseline_2_aggressive_retrofit",
        "proposed_refined_knee",
    ]
    assert decoded_vectors[0].s_z == {"z1": 0, "z2": 0}
    assert decoded_vectors[0].cap_ac_new == {"z1": 0.0, "z2": 0.0}
    assert decoded_vectors[0].cap_rdhx == {"z1": 0.0, "z2": 0.0}
    assert decoded_vectors[0].cap_cdu == {"z1": 0.0, "z2": 0.0}
    assert decoded_vectors[0].cap_ashp == 0.0
    assert decoded_vectors[0].cap_bess == 0.0
    assert decoded_vectors[0].delta_cap_chiller == 0.0
    assert decoded_vectors[0].delta_cap_tower == 0.0
    assert decoded_vectors[1].s_z == {"z1": 6, "z2": 6}
    assert decoded_vectors[1].cap_ac_new["z1"] >= peak_by_zone["z1"]
    assert decoded_vectors[1].cap_cdu["z2"] == pytest.approx(0.7 * peak_by_zone["z2"])
    assert decoded_vectors[1].cap_ashp >= time_frame["heating_demand_kw"].max()
    assert decoded_vectors[1].cap_wshp >= time_frame["heating_demand_kw"].max()
    assert decoded_vectors[2].s_z == {"z1": 2, "z2": 4}
    assert comparison.loc[2, "source_solution_id"] == 13
    assert comparison.loc[2, "tlcc"] == 1006.0
    assert comparison.loc[2, "tce"] == 2006.0
    assert comparison.loc[2, "annual_it_energy_kwh"] == 1100.0
    assert comparison.loc[2, "annual_facility_energy_kwh"] == 1320.0
    assert comparison.loc[2, "annual_cooling_energy_kwh"] == 220.0
    assert comparison.loc[2, "annual_heat_supply_kwh"] == 280.0
    assert comparison.loc[2, "waste_heat_recovery_rate"] == 280.0 / 1100.0
    assert comparison.loc[2, "operational_cost_yuan"] == (120.0 * 1.0 - 20.0 * 0.1) * 2.0 + (
        360.0 * 2.0 - 80.0 * 0.1
    ) * 3.0
    assert comparison.loc[2, "operational_carbon_kg"] == (120.0 * 0.5 - 20.0 * 0.2) * 2.0 + (
        360.0 * 0.6 - 80.0 * 0.2
    ) * 3.0
    assert comparison.loc[2, "pue"] == 1.2
    assert "fixed_cost_payback_years" in comparison.columns
    assert "embodied_carbon_payback_years" in comparison.columns


def test_benchmark_payback_metrics_use_baseline_operational_savings():
    frame = pd.DataFrame(
        {
            "scenario_key": ["baseline_1_no_retrofit", "baseline_2_aggressive_retrofit"],
            "tlcc": [1000.0, 1200.0],
            "tce": [500.0, 550.0],
            "operational_cost_yuan": [900.0, 800.0],
            "operational_carbon_kg": [400.0, 350.0],
        }
    )

    output = _add_payback_metrics(frame)

    assert output.loc[0, "fixed_cost_increment_vs_baseline_yuan_per_year"] == 0.0
    assert output.loc[0, "embodied_carbon_increment_vs_baseline_kg_per_year"] == 0.0
    assert output.loc[0, "fixed_cost_payback_years"] == 0.0
    assert output.loc[0, "embodied_carbon_payback_years"] == 0.0
    assert output.loc[1, "fixed_cost_component_yuan_per_year"] == 400.0
    assert output.loc[1, "embodied_carbon_component_kg_per_year"] == 200.0
    assert output.loc[1, "operational_cost_saving_vs_baseline_yuan_per_year"] == 100.0
    assert output.loc[1, "operational_carbon_saving_vs_baseline_kg_per_year"] == 50.0
    assert output.loc[1, "fixed_cost_payback_years"] == 3.0
    assert output.loc[1, "embodied_carbon_payback_years"] == 2.0
