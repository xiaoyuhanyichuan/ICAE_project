import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_optimization
from analyze_results import write_report
from model import EvaluationResult
from plotting import pareto_distribution_frame, save_pareto_plot


def test_write_report_summarizes_results_assumptions_and_convergence(monkeypatch):
    written = {}

    def capture_write(path, text, encoding=None):
        written["path"] = path
        written["text"] = text
        written["encoding"] = encoding
        return len(text)

    convergence_summary = {
        "available": True,
        "indicator_backend": "fallback_2d",
        "reference_front": "history_nondominated",
        "final_generation": 3,
        "recommended_generation": 2,
        "final_hv": 1.2,
        "final_igd_plus": 0.01,
        "final_n_feasible": 2,
        "final_n_front": 2,
        "hv_relative_tolerance": 0.005,
        "igd_plus_relative_tolerance": 0.005,
        "patience_generations": 5,
    }

    def fake_exists(path):
        return path.name == "pymoo_convergence_summary.json"

    def fake_read_text(path, encoding=None):
        return json.dumps(convergence_summary)

    monkeypatch.setattr(Path, "write_text", capture_write)
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "read_text", fake_read_text)

    all_evaluations = pd.DataFrame(
        {
            "generation": [0, 1, 2],
            "feasible": [True, False, True],
            "tlcc": [120.0, 90.0, 100.0],
            "tce": [45.0, 60.0, 40.0],
        }
    )
    pareto = all_evaluations.iloc[[0, 2]].copy()
    config = {
        "assumptions": {
            "economic_defaults": {
                "source": "assumed",
                "confidence": "medium",
                "description": "Used until vendor quotations are available.",
            }
        }
    }

    report_path = write_report(all_evaluations, pareto, config, "out")

    assert report_path == Path("out") / "report.md"
    assert written["path"] == report_path
    assert written["encoding"] == "utf-8"
    report = written["text"]
    assert "# 优化结果报告" in report
    assert "总评估方案数：3" in report
    assert "可行方案数：2" in report
    assert "Pareto 方案数：2" in report
    assert "Pareto 年化成本范围：100.00 yuan/year - 120.00 yuan/year" in report
    assert "Pareto 年化碳排范围：40.00 kgCO2/year - 45.00 kgCO2/year" in report
    assert "指标后端：fallback_2d" in report
    assert "建议迭代到第 2 代附近后复核" in report
    assert "`economic_defaults`：source=assumed，confidence=medium，Used until vendor quotations are available." in report


def test_write_report_handles_empty_pareto(monkeypatch):
    written = {}

    def capture_write(path, text, encoding=None):
        written["text"] = text
        return len(text)

    monkeypatch.setattr(Path, "write_text", capture_write)
    monkeypatch.setattr(Path, "exists", lambda path: False)
    all_evaluations = pd.DataFrame({"feasible": [False], "tlcc": [1.0], "tce": [2.0]})
    pareto = all_evaluations.iloc[0:0].copy()

    write_report(all_evaluations, pareto, {"assumptions": {}}, "out")

    report = written["text"]
    assert "Pareto 方案数：0" in report
    assert "Pareto 年化成本范围：N/A - N/A" in report
    assert "Pareto 年化碳排范围：N/A - N/A" in report
    assert "收敛性指标暂不可用" in report


def test_write_report_includes_spatial_visualization_section(monkeypatch):
    written = {}

    def capture_write(path, text, encoding=None):
        written["text"] = text
        return len(text)

    def fake_exists(path):
        return path.name in {
            "knee_solution_rack_layout.csv",
            "config_choice_relationship.csv",
            "room_layout_knee_solution_room01.png",
            "config_choice_vs_growth_age.png",
        }

    def fake_glob(path, pattern):
        if pattern == "room_layout_knee_solution_room*.png":
            return [path / "room_layout_knee_solution_room01.png"]
        return []

    def fake_read_csv(path, *args, **kwargs):
        del args, kwargs
        if Path(path).name == "config_choice_relationship.csv":
            return pd.DataFrame(
                {
                    "ac_unit": ["cdz1"],
                    "config_name": ["cold_plate_cdu_wshp"],
                    "rho_load_mean": [0.12],
                    "alpha_age": [0.20],
                    "zone_peak_it_kw": [88.0],
                }
            )
        raise AssertionError(f"unexpected csv read: {path}")

    monkeypatch.setattr(Path, "write_text", capture_write)
    monkeypatch.setattr(Path, "exists", fake_exists)
    monkeypatch.setattr(Path, "glob", fake_glob)
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    all_evaluations = pd.DataFrame({"feasible": [True], "tlcc": [1.0], "tce": [2.0]})
    pareto = all_evaluations.copy()

    write_report(all_evaluations, pareto, {"assumptions": {}}, "out")

    report = written["text"]
    assert "## 7. 机房空间可视化" in report
    assert "room_layout_knee_solution_room01.png" in report
    assert "config_choice_vs_growth_age.png" in report
    assert "cdz1：构型=cold_plate_cdu_wshp" in report
    assert "平均业务爬升率=12.00%" in report
    assert "空调老化率=20.00%" in report


def test_save_pareto_plot_writes_png_under_figures(monkeypatch):
    calls = {}

    class FakeAxis:
        def scatter(self, x_values, y_values):
            calls["scatter"] = (list(x_values), list(y_values))

        def set_xlabel(self, value):
            calls["xlabel"] = value

        def set_ylabel(self, value):
            calls["ylabel"] = value

        def set_title(self, value):
            calls["title"] = value

    class FakeFigure:
        def tight_layout(self):
            calls["tight_layout"] = True

        def savefig(self, path, dpi=None):
            calls["savefig"] = (path, dpi)

    import matplotlib.pyplot as plt

    monkeypatch.setattr(Path, "mkdir", lambda path, parents=False, exist_ok=False: calls.setdefault("mkdir", (path, parents, exist_ok)))
    monkeypatch.setattr(plt, "subplots", lambda: (FakeFigure(), FakeAxis()))
    monkeypatch.setattr(plt, "close", lambda fig: calls.__setitem__("closed", fig))
    pareto = pd.DataFrame({"tlcc": [100.0, 120.0], "tce": [50.0, 40.0]})

    plot_path = save_pareto_plot(pareto, "out")

    assert plot_path == Path("out") / "figures" / "pareto.png"
    assert calls["mkdir"] == (Path("out") / "figures", True, True)
    assert calls["scatter"] == ([100.0, 120.0], [50.0, 40.0])
    assert calls["xlabel"] == "TLCC (yuan/year)"
    assert calls["ylabel"] == "TCE (kgCO2/year)"
    assert calls["title"] == "Pareto Front"
    assert calls["tight_layout"] is True
    assert calls["savefig"] == (plot_path, 150)


def test_pareto_distribution_frame_flattens_decision_json():
    pareto = pd.DataFrame(
        {
            "config_by_zone_json": [json.dumps({"z1": 1, "z2": 4, "z3": 4})],
            "zone_capacity_json": [
                json.dumps(
                    {
                        "cap_ac_new": {"z1": 100.0, "z2": 0.0, "z3": 0.0},
                        "cap_cdu": {"z1": 0.0, "z2": 50.0, "z3": 60.0},
                    }
                )
            ],
            "system_capacity_json": [json.dumps({"cap_wshp": 200.0, "cap_bess": 30.0})],
        }
    )
    config = {"technology": {"configurations": {"1": "new_ac", "4": "cold_plate_cdu_wshp"}}}

    frame = pareto_distribution_frame(pareto, config)

    assert frame.loc[0, "n_new_ac"] == 1.0
    assert frame.loc[0, "n_cold_plate_cdu_wshp"] == 2.0
    assert frame.loc[0, "cap_ac_new_kw"] == 100.0
    assert frame.loc[0, "cap_cdu_kw"] == 110.0
    assert frame.loc[0, "cap_wshp_kw"] == 200.0
    assert frame.loc[0, "cap_bess_kwh"] == 30.0


def test_run_exports_empty_pareto_without_plot_when_all_solutions_infeasible(monkeypatch):
    csv_writes = []
    config = {
        "paths": {"output_dir": "out"},
        "solver": {"nsga2": {"population_size": 2, "random_seed": 42}},
        "typical_days": {},
        "assumptions": {},
    }
    data = SimpleNamespace(ac_hourly=pd.DataFrame({"ac_unit": ["z1"]}), hourly=pd.DataFrame())
    time_index = SimpleNamespace(frame=pd.DataFrame({"it_load_kw": [10.0]}))

    class FakeSchema:
        def __init__(self, zone_ids):
            self.zone_ids = zone_ids

        def random_vector(self, rng):
            return np.array([1.0])

    results = iter(
        [
            EvaluationResult(False, 100.0, 80.0, violation=2.0),
            EvaluationResult(False, 90.0, 70.0, violation=1.0),
        ]
    )

    def fail_if_plot_called(pareto, output_dir):
        raise AssertionError("plot should not be saved for an empty feasible Pareto set")

    monkeypatch.setattr(run_optimization, "load_config", lambda path: config)
    monkeypatch.setattr(run_optimization, "load_input_data", lambda loaded_config: data)
    monkeypatch.setattr(run_optimization, "build_time_index", lambda hourly, typical_days: time_index)
    monkeypatch.setattr(run_optimization, "DecisionSchema", FakeSchema)
    monkeypatch.setattr(run_optimization, "evaluate_solution", lambda *args, **kwargs: next(results))
    monkeypatch.setattr(run_optimization, "save_pareto_plot", fail_if_plot_called)
    monkeypatch.setattr(run_optimization, "analyze_pymoo_convergence", lambda *args: {"summary": {}, "metrics": pd.DataFrame()})
    monkeypatch.setattr(run_optimization, "save_convergence_plots", lambda *args: {})
    monkeypatch.setattr(run_optimization, "save_pareto_distribution_plot", lambda *args: None)
    monkeypatch.setattr(run_optimization, "run_spatial_analysis", lambda *args, **kwargs: {"available": True})
    monkeypatch.setattr(run_optimization, "_resolve_output_dir", lambda loaded_config, config_file: Path("out"))
    monkeypatch.setattr(run_optimization, "write_report", lambda *args: Path("out") / "report.md")
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda frame, path, *args, **kwargs: csv_writes.append((Path(path), frame.copy(), kwargs)))

    result = run_optimization.run("config.json")

    assert result["pareto"].empty
    assert result["figure_path"] is None
    assert result["distribution_path"] is None
    assert result["spatial_result"] == {"available": False, "reason": "empty_pareto"}
    assert result["report_path"] == Path("out") / "report.md"
    assert csv_writes[0][0] == Path("out") / "all_evaluations.csv"
    assert csv_writes[1][0] == Path("out") / "pareto_solutions.csv"
    assert csv_writes[1][1].empty


def test_run_evaluates_initial_population_and_each_generation(monkeypatch):
    config = {
        "paths": {"output_dir": "out"},
        "solver": {
            "nsga2": {
                "population_size": 4,
                "generations": 3,
                "crossover_probability": 1.0,
                "mutation_probability": 0.0,
                "random_seed": 42,
            }
        },
        "typical_days": {},
        "assumptions": {},
    }
    data = SimpleNamespace(ac_hourly=pd.DataFrame({"ac_unit": ["z1"]}), hourly=pd.DataFrame())
    time_index = SimpleNamespace(frame=pd.DataFrame({"it_load_kw": [10.0]}))
    evaluated_vectors = []

    class FakeSchema:
        def __init__(self, zone_ids):
            self.zone_ids = zone_ids
            self.vector_length = 1
            self.next_value = 0

        def random_vector(self, rng):
            self.next_value += 1
            return np.array([float(self.next_value)])

        def bounds(self):
            return np.array([0.0]), np.array([100.0])

    def fake_evaluate(vector, *args, **kwargs):
        evaluated_vectors.append(float(vector[0]))
        return EvaluationResult(True, float(vector[0]), 100.0 - float(vector[0]))

    monkeypatch.setattr(run_optimization, "load_config", lambda path: config)
    monkeypatch.setattr(run_optimization, "load_input_data", lambda loaded_config: data)
    monkeypatch.setattr(run_optimization, "build_time_index", lambda hourly, typical_days: time_index)
    monkeypatch.setattr(run_optimization, "DecisionSchema", FakeSchema)
    monkeypatch.setattr(run_optimization, "evaluate_solution", fake_evaluate)
    monkeypatch.setattr(run_optimization, "save_pareto_plot", lambda pareto, output_dir: Path(output_dir) / "fig.png")
    monkeypatch.setattr(run_optimization, "analyze_pymoo_convergence", lambda *args: {"summary": {}, "metrics": pd.DataFrame()})
    monkeypatch.setattr(run_optimization, "save_convergence_plots", lambda *args: {})
    monkeypatch.setattr(run_optimization, "save_pareto_distribution_plot", lambda *args: Path("out") / "dist.png")
    monkeypatch.setattr(run_optimization, "run_spatial_analysis", lambda *args, **kwargs: {"available": True, "solution_id": 1})
    monkeypatch.setattr(run_optimization, "_resolve_output_dir", lambda loaded_config, config_file: Path("out"))
    monkeypatch.setattr(run_optimization, "write_report", lambda *args: Path("out") / "report.md")
    monkeypatch.setattr(run_optimization, "_write_pareto_artifacts", lambda *args: [], raising=False)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

    result = run_optimization.run("config.json")

    assert len(evaluated_vectors) == 4 * (1 + 3)
    assert len(result["all_evaluations"]) == 4 * (1 + 3)
    assert len(result["pareto"]) == 4 * (1 + 3)
    assert result["distribution_path"] == Path("out") / "dist.png"
    assert result["spatial_result"] == {"available": True, "solution_id": 1}


def test_write_pareto_artifacts_saves_dispatch_csv_and_solution_json(monkeypatch):
    output_dir = Path("out")
    writes = {}
    made_dirs = []

    monkeypatch.setattr(Path, "mkdir", lambda path, parents=False, exist_ok=False: made_dirs.append(path))
    monkeypatch.setattr(Path, "write_text", lambda path, text, encoding=None: writes.setdefault(path, text))
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda frame, path, *args, **kwargs: writes.setdefault(Path(path), frame.copy()))

    schema = run_optimization.DecisionSchema(["z1"])
    vector = schema.encode_default()
    vector[0] = 4
    vector[4] = 10.0
    decision = schema.decode(vector)
    dispatch = pd.DataFrame({"timestamp_hour_utc": [0], "q_cp_kw": [10.0], "q_heat_kw": [3.0]})
    result = EvaluationResult(
        True,
        12.0,
        5.0,
        reasons=["kept"],
        artifacts={
            "decision": decision,
            "dispatch": dispatch,
            "repaired_vector": vector,
            "repair_actions": ["z1:ok"],
            "diagnostics": {"status": "ok"},
        },
    )
    pareto = pd.DataFrame({"solution_id": [7], "tlcc": [12.0], "tce": [5.0], "feasible": [True]})

    written = run_optimization._write_pareto_artifacts(output_dir, pareto, {7: vector}, {7: result})

    solution_json = Path(written[0]["solution_json_path"])
    dispatch_csv = Path(written[0]["dispatch_path"])
    payload = json.loads(writes[solution_json])

    assert written == [{"solution_id": 7, "solution_json_path": solution_json, "dispatch_path": dispatch_csv}]
    assert solution_json.parent.parent == output_dir / "pareto_details"
    assert solution_json.parent in made_dirs
    assert payload["solution_id"] == 7
    assert payload["objectives"] == {"tlcc": 12.0, "tce": 5.0}
    assert payload["decision"]["s_z"] == {"z1": 4}
    assert payload["reasons"] == ["kept"]
    assert writes[dispatch_csv]["q_cp_kw"].tolist() == [10.0]
