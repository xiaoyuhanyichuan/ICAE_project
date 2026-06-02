import json
from pathlib import Path

import pandas as pd

from convergence_analysis import analyze_pymoo_convergence


def test_analyze_pymoo_convergence_writes_metrics_and_summary(monkeypatch):
    writes = {}

    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda frame, path, *args, **kwargs: writes.setdefault(Path(path), frame.copy()))
    monkeypatch.setattr(Path, "write_text", lambda path, text, encoding=None: writes.setdefault(Path(path), text))

    all_evaluations = pd.DataFrame(
        {
            "generation": [0, 0, 1, 1, 2, 2, 3],
            "feasible": [True, True, True, False, True, True, True],
            "tlcc": [100.0, 120.0, 90.0, 80.0, 88.0, 130.0, 87.0],
            "tce": [50.0, 40.0, 48.0, 30.0, 42.0, 35.0, 41.0],
        }
    )

    result = analyze_pymoo_convergence(
        all_evaluations,
        "out",
        {
            "convergence": {
                "patience_generations": 2,
                "min_generation": 1,
                "hv_relative_tolerance": 0.20,
                "igd_plus_relative_tolerance": 0.20,
            }
        },
    )

    metrics = result["metrics"]
    summary = result["summary"]
    metrics_path = Path("out") / "convergence" / "pymoo_convergence_metrics.csv"
    summary_path = Path("out") / "convergence" / "pymoo_convergence_summary.json"

    assert summary["available"] is True
    assert summary["reference_front"] == "history_nondominated"
    assert summary["final_generation"] == 3
    assert summary["final_n_feasible"] == 6
    assert summary["n_generation_metrics"] == 4
    assert summary["observed_generations"] == [0, 1, 2, 3]
    assert summary["final_hv"] == metrics["hv"].iloc[-1]
    assert set(["generation", "hv", "igd_plus", "n_front"]).issubset(metrics.columns)
    assert writes[metrics_path].equals(metrics)
    payload = json.loads(writes[summary_path])
    assert payload["available"] is True


def test_analyze_pymoo_convergence_handles_no_feasible_solutions(monkeypatch):
    writes = {}

    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda frame, path, *args, **kwargs: writes.setdefault(Path(path), frame.copy()))
    monkeypatch.setattr(Path, "write_text", lambda path, text, encoding=None: writes.setdefault(Path(path), text))

    all_evaluations = pd.DataFrame(
        {
            "generation": [0, 1],
            "feasible": [False, False],
            "tlcc": [float("inf"), float("inf")],
            "tce": [float("inf"), float("inf")],
        }
    )

    result = analyze_pymoo_convergence(all_evaluations, "out", {})
    summary_path = Path("out") / "convergence" / "pymoo_convergence_summary.json"

    assert result["summary"]["available"] is False
    assert result["summary"]["reason"] == "no_feasible_finite_solutions"
    assert result["metrics"].empty
    assert json.loads(writes[summary_path])["available"] is False


def test_config_requests_extended_convergence_horizon():
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))

    assert int(config["solver"]["nsga2"]["generations"]) >= 80
    assert int(config["convergence"]["min_generation"]) >= 20
    assert int(config["convergence"]["patience_generations"]) >= 10
