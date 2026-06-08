import numpy as np
import pandas as pd
import pytest

from diagnose_pareto_scarcity import (
    DiagnosticScenario,
    _build_report,
    _make_diagnostic_scenarios,
    _markdown_table,
    _summarize_scenario,
)
from feasibility_oracle import ScreeningResult
from model import EvaluationResult


def _base_config():
    return {
        "paths": {"output_dir": "results"},
        "solver": {"infeasible_memory": True},
        "decision_bounds": {
            "default_zone_cap_upper_kw": 100.0,
            "default_system_cap_upper_kw": 200.0,
            "zone_cap_upper_kw": {"cap_ac_new": 10.0, "cap_rdhx": 20.0, "cap_cdu": 30.0},
            "system_cap_upper_kw": {"cap_ashp": 40.0, "cap_wshp": 50.0},
        },
        "scenario": {
            "existing_cooling_redundancy": {
                "old_ac_factor": 1.05,
                "chiller_factor": 1.05,
                "tower_factor": 1.05,
            }
        },
        "weight": {
            "hard_constraint": {
                "enabled": True,
                "abs_tolerance_kg": 1.0e-6,
                "relative_tolerance": 1.0e-9,
            }
        },
    }


def test_diagnostic_scenarios_apply_requested_config_ablations():
    scenarios = {scenario.name: scenario for scenario in _make_diagnostic_scenarios(_base_config())}

    assert scenarios["S1_bypass_screen"].config["solver"]["screening"]["bypass_hard"] is True
    assert scenarios["S1_bypass_screen"].config["solver"]["infeasible_memory"] is False
    assert scenarios["S2_no_infeasible_memory"].config["solver"]["infeasible_memory"] is False
    assert scenarios["S3_no_zone_peak_screen"].config["solver"]["screening"]["disabled_hard_checks"] == [
        "zone_peak"
    ]
    assert scenarios["P1_capacity_relaxed"].config["decision_bounds"]["zone_cap_upper_kw"]["cap_cdu"] == 45.0
    assert scenarios["P1_capacity_relaxed"].config["decision_bounds"]["system_cap_upper_kw"]["cap_wshp"] == 75.0
    assert (
        scenarios["P2_redundancy_relaxed"].config["scenario"]["existing_cooling_redundancy"][
            "old_ac_factor"
        ]
        == 1.20
    )
    legacy_config_key = "heat" + "_demand" + "_scaling"
    assert all(not name.startswith("P3_") for name in scenarios)
    assert all(legacy_config_key not in scenario.config.get("scenario", {}) for scenario in scenarios.values())
    assert all(not name.startswith("O") for name in scenarios)


def test_scenario_summary_counts_pareto_and_screening_rejected_feasible_cases():
    scenario = DiagnosticScenario("S1_bypass_screen", "screening", "test", {})
    candidates = [np.array([idx], dtype=float) for idx in range(3)]
    results = [
        EvaluationResult(
            feasible=True,
            tlcc=10.0,
            tce=30.0,
            artifacts={"screening": ScreeningResult(feasible=True)},
        ),
        EvaluationResult(
            feasible=True,
            tlcc=20.0,
            tce=20.0,
            artifacts={"screening": ScreeningResult(feasible=False, reasons=["zone_peak_shortage"])},
        ),
        EvaluationResult(
            feasible=False,
            tlcc=float("inf"),
            tce=float("inf"),
            reasons=["inner_dispatch_infeasible"],
            artifacts={"screening": ScreeningResult(feasible=True)},
        ),
    ]

    rows, summary = _summarize_scenario(scenario, candidates, results)

    assert len(rows) == 3
    assert summary["milp_feasible_count"] == 2
    assert summary["screen_reject_count"] == 1
    assert summary["screen_rejected_milp_feasible_count"] == 1
    assert summary["inner_infeasible_count"] == 1
    assert summary["pareto_count"] == 2


def test_report_builder_outputs_readable_chinese_without_tabulate_dependency():
    summary = pd.DataFrame(
        [
            {
                "scenario": "S0_current",
                "screen_reject_count": 0,
                "milp_feasible_count": 10,
                "inner_infeasible_count": 1,
                "pareto_count": 2,
                "screen_rejected_milp_feasible_rate": 0.0,
                "mean_weight_hard_violation": 0.0,
                "candidate_count": 11,
                "top_infeasible_reasons": "inner_dispatch_infeasible=1",
                "top_screening_reasons": "",
            },
            {
                "scenario": "S1_bypass_screen",
                "screen_reject_count": 0,
                "milp_feasible_count": 10,
                "inner_infeasible_count": 1,
                "pareto_count": 2,
                "screen_rejected_milp_feasible_rate": 0.0,
                "mean_weight_hard_violation": 0.0,
                "candidate_count": 11,
                "top_infeasible_reasons": "inner_dispatch_infeasible=1",
                "top_screening_reasons": "",
            },
        ]
    )

    report = _build_report(summary, evaluations_path="results/all_evaluations.csv", candidate_count=11, plot_paths={})

    assert "帕累托前沿解数过少诊断报告" in report
    assert "场景汇总" in report
    assert "预筛查不是主要原因" in report
    assert "| scenario |" in _markdown_table(summary[["scenario", "pareto_count"]])
