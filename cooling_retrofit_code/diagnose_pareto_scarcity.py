from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_utils import load_config, load_input_data
from decision import DecisionSchema
from model import EvaluationResult
from nsga2 import fast_non_dominated_sort
from run_optimization import PopulationEvaluator, infer_zone_ids, peak_load_by_zone_from_data
from typical_days import build_time_index


@dataclass(frozen=True)
class DiagnosticScenario:
    name: str
    category: str
    description: str
    config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why the Pareto front has few solutions.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.json"),
        help="Path to the optimization config JSON.",
    )
    parser.add_argument(
        "--evaluations",
        default=None,
        help="Path to all_evaluations.csv. Defaults to config paths.output_dir/all_evaluations.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostic outputs. Defaults to results/diagnostics/pareto_scarcity.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional deterministic sample size. Use 0 to evaluate all candidates.",
    )
    parser.add_argument("--seed", type=int, default=20260604, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    base_config = load_config(config_path)
    evaluations_path = _resolve_evaluations_path(base_config, config_path, args.evaluations)
    output_dir = _resolve_output_dir(base_config, config_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidate_vectors(evaluations_path, args.max_candidates, args.seed)
    scenarios = _make_diagnostic_scenarios(base_config)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        print(f"[pareto-diagnostic] evaluating {scenario.name} with {len(candidates)} candidates")
        results = _evaluate_candidates(scenario.config, candidates)
        evaluation_rows, summary = _summarize_scenario(scenario, candidates, results)
        all_rows.extend(evaluation_rows)
        summary_rows.append(summary)

    evaluations = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows)
    evaluations.to_csv(output_dir / "pareto_scarcity_evaluations.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "pareto_scarcity_scenario_summary.csv", index=False, encoding="utf-8-sig")

    plot_paths = _save_diagnostic_plots(evaluations, summary, output_dir)
    report_path = output_dir / "pareto_scarcity_report.md"
    report_path.write_text(
        _build_report(summary, evaluations_path, len(candidates), plot_paths),
        encoding="utf-8",
    )
    print(f"[pareto-diagnostic] wrote {report_path}")


def _resolve_evaluations_path(base_config: dict[str, Any], config_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    output_dir = Path(base_config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    return output_dir / "all_evaluations.csv"


def _resolve_output_dir(base_config: dict[str, Any], config_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    output_dir = Path(base_config["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    return output_dir / "diagnostics" / "pareto_scarcity"


def _load_candidate_vectors(path: Path, max_candidates: int, seed: int) -> list[np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate source: {path}")
    frame = pd.read_csv(path)
    if "raw_vector_json" not in frame.columns:
        raise ValueError(f"{path} must contain raw_vector_json")
    selected = _select_candidate_rows(frame, max_candidates, seed)
    vectors: list[np.ndarray] = []
    for text in selected["raw_vector_json"].fillna(""):
        values = json.loads(str(text))
        vectors.append(np.asarray(values, dtype=float))
    return vectors


def _select_candidate_rows(frame: pd.DataFrame, max_candidates: int, seed: int) -> pd.DataFrame:
    if max_candidates <= 0 or len(frame) <= max_candidates:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    priority = frame[
        (pd.to_numeric(frame.get("front", pd.Series(np.nan, index=frame.index)), errors="coerce") == 0)
        | (~frame.get("feasible", pd.Series(True, index=frame.index)).astype(bool))
    ]
    priority_indices = priority.index.tolist()
    remaining_slots = max(0, max_candidates - len(priority_indices))
    remaining = frame.drop(index=priority_indices, errors="ignore")
    if remaining_slots > 0 and not remaining.empty:
        sample_size = min(remaining_slots, len(remaining))
        sampled_indices = rng.choice(remaining.index.to_numpy(), size=sample_size, replace=False).tolist()
    else:
        sampled_indices = []
    keep = priority_indices + sampled_indices
    return frame.loc[keep[:max_candidates]].sort_index().reset_index(drop=True)


def _make_diagnostic_scenarios(base_config: dict[str, Any]) -> list[DiagnosticScenario]:
    scenarios: list[DiagnosticScenario] = []
    scenarios.append(
        DiagnosticScenario(
            "S0_current",
            "screening",
            "Current repair + screening + MILP.",
            deepcopy(base_config),
        )
    )
    scenarios.append(
        DiagnosticScenario(
            "S1_bypass_screen",
            "screening",
            "Repair, record screening, then bypass hard screening and solve MILP.",
            _with_screening(base_config, bypass_hard=True, infeasible_memory=False),
        )
    )
    scenarios.append(
        DiagnosticScenario(
            "S2_no_infeasible_memory",
            "screening",
            "Current screening with infeasible-memory cache disabled.",
            _with_screening(base_config, bypass_hard=False, infeasible_memory=False),
        )
    )
    for check_name, label in (
        ("zone_peak", "S3_no_zone_peak_screen"),
        ("chiller_peak", "S4_no_chiller_peak_screen"),
        ("tower_peak", "S5_no_tower_peak_screen"),
    ):
        scenarios.append(
            DiagnosticScenario(
                label,
                "screening_component",
                f"Disable only the {check_name} hard screening check.",
                _with_disabled_hard_check(base_config, check_name),
            )
        )
    scenarios.append(
        DiagnosticScenario(
            "P1_capacity_relaxed",
            "parameter",
            "Increase zone and system capacity upper bounds by 50%.",
            _with_capacity_bounds_multiplier(base_config, 1.5),
        )
    )
    scenarios.append(
        DiagnosticScenario(
            "P2_redundancy_relaxed",
            "parameter",
            "Raise old AC/chiller/tower redundancy factors to 1.20.",
            _with_redundancy(base_config, 1.20),
        )
    )
    return scenarios


def _with_screening(
    base_config: dict[str, Any],
    *,
    bypass_hard: bool,
    infeasible_memory: bool,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    solver = config.setdefault("solver", {})
    solver["infeasible_memory"] = bool(infeasible_memory)
    screening = solver.setdefault("screening", {})
    screening["bypass_hard"] = bool(bypass_hard)
    return config


def _with_disabled_hard_check(base_config: dict[str, Any], check_name: str) -> dict[str, Any]:
    config = _with_screening(base_config, bypass_hard=False, infeasible_memory=False)
    config.setdefault("solver", {}).setdefault("screening", {})["disabled_hard_checks"] = [check_name]
    return config


def _with_capacity_bounds_multiplier(base_config: dict[str, Any], multiplier: float) -> dict[str, Any]:
    config = deepcopy(base_config)
    bounds = config.setdefault("decision_bounds", {})
    for key in ("default_zone_cap_upper_kw", "default_system_cap_upper_kw"):
        if key in bounds:
            bounds[key] = float(bounds[key]) * multiplier
    for key in ("zone_cap_upper_kw", "system_cap_upper_kw"):
        bounds[key] = _scale_bound(bounds.get(key, {}), multiplier)
    return config


def _scale_bound(value: Any, multiplier: float) -> Any:
    if isinstance(value, dict):
        return {name: float(bound) * multiplier for name, bound in value.items()}
    if value == {}:
        return value
    return float(value) * multiplier


def _with_redundancy(base_config: dict[str, Any], factor: float) -> dict[str, Any]:
    config = deepcopy(base_config)
    redundancy = config.setdefault("scenario", {}).setdefault("existing_cooling_redundancy", {})
    redundancy["old_ac_factor"] = factor
    redundancy["chiller_factor"] = factor
    redundancy["tower_factor"] = factor
    return config


def _evaluate_candidates(config: dict[str, Any], candidates: list[np.ndarray]) -> list[EvaluationResult]:
    data = load_input_data(config)
    time_index = build_time_index(data.hourly, config["typical_days"])
    zone_ids = infer_zone_ids(data)
    schema = DecisionSchema.from_config(zone_ids, config)
    peak_load_by_zone_kw = peak_load_by_zone_from_data(
        time_index.frame,
        getattr(data, "rack_metadata", pd.DataFrame()),
        zone_ids,
    )
    with PopulationEvaluator(schema, config, time_index.frame, peak_load_by_zone_kw) as evaluator:
        return evaluator.evaluate(candidates)


def _summarize_scenario(
    scenario: DiagnosticScenario,
    candidates: list[np.ndarray],
    results: list[EvaluationResult],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    feasible_positions = [idx for idx, result in enumerate(results) if bool(result.feasible)]
    feasible_results = [results[idx] for idx in feasible_positions]
    pareto_positions: set[int] = set()
    if feasible_results:
        fronts = fast_non_dominated_sort(feasible_results)
        if fronts:
            pareto_positions = {feasible_positions[idx] for idx in fronts[0]}

    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        artifacts = result.artifacts if isinstance(result.artifacts, dict) else {}
        screening = artifacts.get("screening")
        diagnostics = artifacts.get("diagnostics", {})
        screening_rejected = not bool(getattr(screening, "feasible", True))
        screen_reasons = ";".join(getattr(screening, "reasons", []))
        rows.append(
            {
                "scenario": scenario.name,
                "category": scenario.category,
                "candidate_id": idx,
                "feasible": bool(result.feasible),
                "pareto": idx in pareto_positions,
                "tlcc": float(result.tlcc),
                "tce": float(result.tce),
                "violation": float(result.violation),
                "reasons": ";".join(result.reasons),
                "screening_rejected": screening_rejected,
                "screening_bypassed": bool(diagnostics.get("screening_bypassed", False)),
                "screening_reasons": screen_reasons,
                "weight_hard_violation_kg": _weight_hard_violation(screening),
                "inner_build_total_s": float(diagnostics.get("inner_build_total_s", 0.0) or 0.0),
                "inner_update_s": float(diagnostics.get("inner_update_s", 0.0) or 0.0),
                "inner_optimize_s": float(diagnostics.get("inner_optimize_s", 0.0) or 0.0),
            }
        )

    frame = pd.DataFrame(rows)
    feasible = frame[frame["feasible"]]
    screen_rejected = frame[frame["screening_rejected"]]
    summary = {
        "scenario": scenario.name,
        "category": scenario.category,
        "description": scenario.description,
        "candidate_count": len(candidates),
        "screen_reject_count": int(frame["screening_rejected"].sum()),
        "screen_reject_rate": _safe_ratio(frame["screening_rejected"].sum(), len(frame)),
        "screen_rejected_milp_feasible_count": int((screen_rejected["feasible"]).sum())
        if not screen_rejected.empty
        else 0,
        "screen_rejected_milp_feasible_rate": _safe_ratio(
            int((screen_rejected["feasible"]).sum()) if not screen_rejected.empty else 0,
            len(screen_rejected),
        ),
        "milp_feasible_count": int(frame["feasible"].sum()),
        "milp_feasible_rate": _safe_ratio(frame["feasible"].sum(), len(frame)),
        "inner_infeasible_count": int((frame["reasons"].str.contains("inner_dispatch_infeasible")).sum()),
        "pareto_count": int(frame["pareto"].sum()),
        "pareto_share": _safe_ratio(frame["pareto"].sum(), max(1, frame["feasible"].sum())),
        "unique_objective_count": int(
            len(set(zip(feasible["tlcc"].round(6), feasible["tce"].round(6)))) if not feasible.empty else 0
        ),
        "tlcc_min": float(feasible["tlcc"].min()) if not feasible.empty else np.nan,
        "tlcc_max": float(feasible["tlcc"].max()) if not feasible.empty else np.nan,
        "tce_min": float(feasible["tce"].min()) if not feasible.empty else np.nan,
        "tce_max": float(feasible["tce"].max()) if not feasible.empty else np.nan,
        "mean_weight_hard_violation": float(feasible["weight_hard_violation_kg"].mean())
        if not feasible.empty
        else 0.0,
        "mean_tlcc": float(feasible["tlcc"].mean()) if not feasible.empty else np.nan,
        "mean_tce": float(feasible["tce"].mean()) if not feasible.empty else np.nan,
        "top_infeasible_reasons": _top_reasons(frame),
        "top_screening_reasons": _top_screening_reasons(frame),
    }
    return rows, summary


def _weight_hard_violation(screening: Any) -> float:
    reasons = getattr(screening, "reasons", []) if screening is not None else []
    total = 0.0
    for reason in reasons:
        parts = str(reason).split(":")
        if len(parts) >= 3 and parts[-2] == "weight_margin_exceeded":
            try:
                total += max(0.0, float(parts[-1]))
            except ValueError:
                continue
    return total


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / denominator


def _top_reasons(frame: pd.DataFrame) -> str:
    reasons = frame.loc[~frame["feasible"], "reasons"].fillna("").replace("", "<empty>")
    if reasons.empty:
        return ""
    return "; ".join(f"{reason}={count}" for reason, count in reasons.value_counts().head(5).items())


def _top_screening_reasons(frame: pd.DataFrame) -> str:
    reasons = frame.loc[frame["screening_rejected"], "screening_reasons"].fillna("").replace("", "<empty>")
    if reasons.empty:
        return ""
    return "; ".join(f"{reason}={count}" for reason, count in reasons.value_counts().head(5).items())


def _save_diagnostic_plots(evaluations: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": str(exc)}

    paths: dict[str, str] = {}
    x = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 1.2), 5))
    ax.bar(x - 0.24, summary["candidate_count"], width=0.18, label="raw/repaired")
    ax.bar(x - 0.08, summary["candidate_count"] - summary["screen_reject_count"], width=0.18, label="screen passed")
    ax.bar(x + 0.08, summary["milp_feasible_count"], width=0.18, label="MILP feasible")
    ax.bar(x + 0.24, summary["pareto_count"], width=0.18, label="Pareto")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["scenario"], rotation=35, ha="right")
    ax.set_ylabel("candidate count")
    ax.set_title("Candidate Flow Diagnostic")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "pareto_scarcity_funnel.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["funnel"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    for scenario in summary["scenario"]:
        subset = evaluations[(evaluations["scenario"] == scenario) & (evaluations["feasible"])]
        pareto = subset[subset["pareto"]]
        if subset.empty:
            continue
        ax.scatter(subset["tlcc"] / 1.0e6, subset["tce"] / 1.0e6, s=8, alpha=0.12)
        ax.scatter(pareto["tlcc"] / 1.0e6, pareto["tce"] / 1.0e6, s=35, label=scenario)
    ax.set_xlabel("TLCC (million yuan/year)")
    ax.set_ylabel("TCE (million kgCO2/year)")
    ax.set_title("Pareto Front Comparison Across Diagnostic Scenarios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "pareto_scarcity_fronts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["fronts"] = str(path)

    return paths


def _build_report(
    summary: pd.DataFrame,
    evaluations_path: Path,
    candidate_count: int,
    plot_paths: dict[str, str],
) -> str:
    summary_table = _markdown_table(
        summary[
            [
                "scenario",
                "screen_reject_count",
                "milp_feasible_count",
                "inner_infeasible_count",
                "pareto_count",
                "screen_rejected_milp_feasible_rate",
                "mean_weight_hard_violation",
            ]
        ]
    )
    lines = [
        "# 帕累托前沿解数过少诊断报告",
        "",
        f"- 候选解来源：`{evaluations_path}`",
        f"- 固定候选解数量：{candidate_count}",
        f"- 诊断场景数量：{len(summary)}",
        "",
        "## 场景汇总",
        "",
        summary_table,
        "",
        "## 判定结论",
        "",
        *_diagnosis_lines(summary),
        "",
        "## 图表",
        "",
    ]
    for name, path in plot_paths.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(["", "## 详细原因分布", ""])
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['scenario']}`: infeasible reasons [{row['top_infeasible_reasons']}], "
            f"screening reasons [{row['top_screening_reasons']}]"
        )
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无数据_"
    headers = [str(column) for column in frame.columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                elif abs(value) >= 1000:
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in rows)
    return "\n".join(lines)


def _diagnosis_lines(summary: pd.DataFrame) -> list[str]:
    rows = {str(row["scenario"]): row for _, row in summary.iterrows()}
    current = rows.get("S0_current")
    if current is None:
        return ["- 未找到 `S0_current`，无法自动判定。"]

    lines: list[str] = []
    bypass = rows.get("S1_bypass_screen")
    if bypass is not None:
        feasible_gain = int(bypass["milp_feasible_count"]) - int(current["milp_feasible_count"])
        pareto_gain = int(bypass["pareto_count"]) - int(current["pareto_count"])
        false_negative_rate = float(bypass["screen_rejected_milp_feasible_rate"])
        if feasible_gain > 0 or pareto_gain > 0 or false_negative_rate > 0.05:
            lines.append(
                "- 预筛查可能过紧：`S1_bypass_screen` 相比当前场景增加了 "
                f"{feasible_gain} 个可行解、{pareto_gain} 个 Pareto 解，"
                f"screen-rejected MILP 可行率为 {false_negative_rate:.2%}。"
            )
        else:
            lines.append("- 预筛查不是主要原因：跳过硬筛查没有显著增加可行解或 Pareto 解。")

    parameter_names = ["P1_capacity_relaxed", "P2_redundancy_relaxed"]
    parameter_rows = [rows[name] for name in parameter_names if name in rows]
    if parameter_rows:
        best_param = max(parameter_rows, key=lambda row: (int(row["pareto_count"]), int(row["milp_feasible_count"])))
        pareto_gain = int(best_param["pareto_count"]) - int(current["pareto_count"])
        feasible_gain = int(best_param["milp_feasible_count"]) - int(current["milp_feasible_count"])
        if pareto_gain > 0 or feasible_gain > max(5, 0.02 * int(current["candidate_count"])):
            lines.append(
                f"- 参数可行域值得关注：`{best_param['scenario']}` 带来 {feasible_gain} "
                f"个可行解增量、{pareto_gain} 个 Pareto 解增量。"
            )
        else:
            lines.append("- 参数上界和冗余容量的放松没有显著扩大 Pareto 前沿。")

    if not lines:
        lines.append("- 当前诊断场景没有显示单一主因，建议继续检查 NSGA-II 多样性和 repair 后个体坍缩。")
    return lines


if __name__ == "__main__":
    main()
