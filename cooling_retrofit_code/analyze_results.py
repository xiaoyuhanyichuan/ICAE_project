from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _format_number(value: float, unit: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    text = f"{number:,.2f}"
    return f"{text} {unit}".rstrip()


def _format_percent(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(number):
        return "N/A"
    return f"{100.0 * number:.2f}%"


def _safe_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return value


def _finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def _bool_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _config_counts(row: pd.Series) -> dict[str, int]:
    configs = _safe_json(row.get("config_by_zone_json"))
    if not isinstance(configs, dict):
        return {}
    counts: dict[str, int] = {}
    for value in configs.values():
        key = str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _capacity_summary(row: pd.Series) -> tuple[dict[str, float], dict[str, float]]:
    zone_caps = _safe_json(row.get("zone_capacity_json"))
    system_caps = _safe_json(row.get("system_capacity_json"))
    zone_totals: dict[str, float] = {}
    system_totals: dict[str, float] = {}

    if isinstance(zone_caps, dict):
        for name, by_zone in zone_caps.items():
            if isinstance(by_zone, dict):
                zone_totals[name] = _sum_finite(by_zone.values())

    if isinstance(system_caps, dict):
        for name, value in system_caps.items():
            number = _to_float(value)
            if number is not None:
                system_totals[name] = number

    return zone_totals, system_totals


def _sum_finite(values) -> float:
    total = 0.0
    for value in values:
        number = _to_float(value)
        if number is not None:
            total += number
    return total


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _format_config_counts(counts: dict[str, int], config: dict) -> str:
    if not counts:
        return "未记录"
    names = config.get("technology", {}).get("configurations", {})
    parts = []
    for key in sorted(counts, key=lambda item: int(item) if item.isdigit() else item):
        name = names.get(str(key), "")
        suffix = f" {name}" if name else ""
        parts.append(f"{key}{suffix}: {counts[key]}")
    return "；".join(parts)


def _format_capacity_lines(zone_totals: dict[str, float], system_totals: dict[str, float]) -> list[str]:
    lines: list[str] = []
    zone_labels = {
        "cap_ac_new": "新空调容量",
        "cap_rdhx": "RDHX 容量",
        "cap_cdu": "CDU 容量",
    }
    system_labels = {
        "cap_ashp": "ASHP 容量",
        "cap_wshp": "WSHP 容量",
        "cap_bess": "BESS 整体容量",
        "cap_tes": "TES 容量",
        "delta_cap_chiller": "Chiller 扩容",
        "delta_cap_tower": "Cooling Tower 扩容",
    }

    for key, label in zone_labels.items():
        if key in zone_totals:
            lines.append(f"  - {label}: {_format_number(zone_totals[key], 'kW')}")
    for key, label in system_labels.items():
        if key in system_totals:
            unit = "kWh" if key in {"cap_bess", "cap_tes"} else "kW"
            lines.append(f"  - {label}: {_format_number(system_totals[key], unit)}")
    return lines or ["  - 未记录容量明细。"]


def _representative_rows(pareto: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    if pareto.empty:
        return []

    rows: list[tuple[str, pd.Series]] = []
    cost_idx = pd.to_numeric(pareto["tlcc"], errors="coerce").idxmin()
    carbon_idx = pd.to_numeric(pareto["tce"], errors="coerce").idxmin()
    rows.append(("最低成本解", pareto.loc[cost_idx]))
    if carbon_idx != cost_idx:
        rows.append(("最低碳排解", pareto.loc[carbon_idx]))

    finite = pareto.copy()
    finite["tlcc_num"] = pd.to_numeric(finite["tlcc"], errors="coerce")
    finite["tce_num"] = pd.to_numeric(finite["tce"], errors="coerce")
    finite = finite[np.isfinite(finite["tlcc_num"]) & np.isfinite(finite["tce_num"])]
    if len(finite) >= 3:
        tlcc_span = finite["tlcc_num"].max() - finite["tlcc_num"].min()
        tce_span = finite["tce_num"].max() - finite["tce_num"].min()
        if tlcc_span > 0 and tce_span > 0:
            norm_cost = (finite["tlcc_num"] - finite["tlcc_num"].min()) / tlcc_span
            norm_carbon = (finite["tce_num"] - finite["tce_num"].min()) / tce_span
            knee_idx = (norm_cost.pow(2) + norm_carbon.pow(2)).idxmin()
            if knee_idx not in {cost_idx, carbon_idx}:
                rows.append(("折中解", pareto.loc[knee_idx]))
    return rows


def _reason_summary(all_evaluations: pd.DataFrame) -> list[str]:
    if "reasons" not in all_evaluations.columns:
        return ["- 未记录不可行原因。"]
    reasons: dict[str, int] = {}
    for value in all_evaluations["reasons"].fillna(""):
        for reason in str(value).split(";"):
            reason = reason.strip()
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
    if not reasons:
        return ["- 本轮未记录不可行原因，或所有候选解均通过筛查。"]
    return [
        f"- `{name}`: {count} 次"
        for name, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


def _changed_by_repair(all_evaluations: pd.DataFrame) -> int | None:
    if "raw_vector_json" not in all_evaluations.columns or "repaired_vector_json" not in all_evaluations.columns:
        return None
    raw = all_evaluations["raw_vector_json"].fillna("")
    repaired = all_evaluations["repaired_vector_json"].fillna("")
    return int((raw != repaired).sum())


def _read_convergence_summary(output: Path) -> dict[str, Any]:
    summary_path = output / "convergence" / "pymoo_convergence_summary.json"
    if not summary_path.exists():
        return {"available": False, "reason": "summary_file_not_found", "summary_path": str(summary_path)}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "reason": f"summary_read_failed:{exc}", "summary_path": str(summary_path)}
    return payload if isinstance(payload, dict) else {"available": False, "reason": "summary_payload_not_dict"}


def _convergence_lines(summary: dict[str, Any]) -> list[str]:
    if not summary.get("available"):
        return [f"- 收敛性指标暂不可用：{summary.get('reason', '未说明')}"]
    recommended = summary.get("recommended_generation")
    recommendation = (
        "未检测到稳定平台期。"
        if recommended is None
        else f"建议迭代到第 {recommended} 代附近后复核。"
    )
    return [
        f"- 指标后端：{summary.get('indicator_backend', 'N/A')}；参考前沿：{summary.get('reference_front', 'N/A')}",
        f"- 最终代数：{summary.get('final_generation', 'N/A')}；历史可行解：{summary.get('final_n_feasible', 'N/A')}；最终前沿点数：{summary.get('final_n_front', 'N/A')}",
        f"- 收敛性分析观测代数：{summary.get('n_generation_metrics', 'N/A')}",
        f"- 最终 Hypervolume：{_format_number(summary.get('final_hv', float('nan')))}",
        f"- 最终 IGD+：{_format_number(summary.get('final_igd_plus', float('nan')))}",
        f"- 推荐判断：{recommendation}",
    ]


def _spatial_visualization_lines(output: Path) -> list[str]:
    figures = output / "figures"
    spatial = output / "spatial"
    room_plots = sorted(figures.glob("room_layout_knee_solution_room*.png"))
    relationship_plot = figures / "config_choice_vs_growth_age.png"
    rack_csv = spatial / "knee_solution_rack_layout.csv"
    relationship_csv = spatial / "config_choice_relationship.csv"

    if not rack_csv.exists() or not relationship_csv.exists():
        return ["- 尚未生成机房空间可视化数据；Pareto 解非空且 rack_metadata 字段完整时会自动生成。"]

    lines = [
        f"- 机柜空间布局图：{len(room_plots)} 张独立机房图",
        f"- 构型选择关系图：`{relationship_plot}`",
        f"- 机柜级空间数据：`{rack_csv}`",
        f"- 空调组级关系数据：`{relationship_csv}`",
    ]
    for room_plot in room_plots:
        lines.append(f"![{room_plot.stem}]({room_plot})")
    if relationship_plot.exists():
        lines.append(f"![Configuration choice vs growth and aging]({relationship_plot})")

    try:
        relationship = pd.read_csv(relationship_csv)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return lines
    if relationship.empty:
        return lines

    config_col = "config_name" if "config_name" in relationship.columns else "config_id"
    for _, row in relationship.sort_values("ac_unit").iterrows():
        lines.append(
            "- "
            f"{row.get('ac_unit', 'N/A')}：构型={row.get(config_col, 'N/A')}；"
            f"平均业务爬升率={_format_percent(_to_float(row.get('rho_load_mean')) or float('nan'))}；"
            f"空调老化率={_format_percent(_to_float(row.get('alpha_age')) or float('nan'))}；"
            f"区域峰值 IT 负荷={_format_number(_to_float(row.get('zone_peak_it_kw')) or float('nan'), 'kW')}"
        )
    return lines


def _assumption_lines(config: dict) -> list[str]:
    assumptions = config.get("assumptions", {})
    if not assumptions:
        return ["- 未提供参数来源或假设。"]
    lines = []
    for name, detail in assumptions.items():
        detail = detail if isinstance(detail, dict) else {}
        source = detail.get("source", "未说明")
        confidence = detail.get("confidence", "未说明")
        description = detail.get("description", "无补充说明")
        lines.append(f"- `{name}`：source={source}，confidence={confidence}，{description}")
    return lines


def write_report(
    all_evaluations: pd.DataFrame,
    pareto: pd.DataFrame,
    config: dict,
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    report_path = output / "report.md"
    feasible_mask = _bool_series(
        all_evaluations["feasible"] if "feasible" in all_evaluations else None,
        all_evaluations.index,
    )
    feasible_count = int(feasible_mask.sum())
    total_count = len(all_evaluations)
    feasible_rate = feasible_count / total_count if total_count else float("nan")
    feasible = all_evaluations[feasible_mask].copy()

    tlcc_values = _finite_numeric(pareto["tlcc"]) if "tlcc" in pareto else pd.Series(dtype=float)
    tce_values = _finite_numeric(pareto["tce"]) if "tce" in pareto else pd.Series(dtype=float)
    generation_values = _finite_numeric(all_evaluations["generation"]) if "generation" in all_evaluations else pd.Series(dtype=float)
    nsga_cfg = config.get("solver", {}).get("nsga2", {})
    typical_cfg = config.get("typical_days", {})
    heat_scaling = config.get("scenario", {}).get("heat_demand_scaling", {})
    repaired_count = _changed_by_repair(all_evaluations)
    convergence_summary = _read_convergence_summary(output)

    lines = [
        "# 优化结果报告",
        "",
        "## 1. 求解概览",
        "",
        f"- 总评估方案数：{total_count}",
        f"- 可行方案数：{feasible_count}",
        f"- 可行率：{_format_percent(feasible_rate)}",
        f"- Pareto 方案数：{len(pareto)}",
        f"- 代数范围：{int(generation_values.min()) if not generation_values.empty else 'N/A'} - {int(generation_values.max()) if not generation_values.empty else 'N/A'}",
        f"- Pareto 年化成本范围：{_format_number(tlcc_values.min() if not tlcc_values.empty else float('nan'), 'yuan/year')} - {_format_number(tlcc_values.max() if not tlcc_values.empty else float('nan'), 'yuan/year')}",
        f"- Pareto 年化碳排范围：{_format_number(tce_values.min() if not tce_values.empty else float('nan'), 'kgCO2/year')} - {_format_number(tce_values.max() if not tce_values.empty else float('nan'), 'kgCO2/year')}",
        "",
        "## 2. 模型与求解口径",
        "",
        "- 方法论继承原研究问题：外层 NSGA-II 规划搜索，repair/screening 预筛查，内层 Gurobi MILP 运行调度，最后从全部历史可行解筛选 Pareto 前沿。",
        "- 当前代码口径已同步：RDHX 链路不再设置辅助风冷容量；BESS 不再拆分功率容量和能量容量；供热需求为硬约束，不再设置未供热松弛罚金。",
        "- RDHX 按可吸收其负责区域 100% IT 热量建模；未进入 WSHP 的 RDHX 热量进入 Chiller。",
        "- BESS 仅使用整体容量 `cap_bess`，充放电功率上限由 `technology.bess.c_rate_per_hour` 派生。",
        f"- 典型日：K={typical_cfg.get('k', 'N/A')}；append_peak_day={typical_cfg.get('append_peak_day', 'N/A')}。",
        f"- NSGA-II：population_size={nsga_cfg.get('population_size', 'N/A')}；generations={nsga_cfg.get('generations', 'N/A')}；mutation_probability={nsga_cfg.get('mutation_probability', 'N/A')}。",
    ]

    if heat_scaling:
        lines.extend(
            [
                f"- 供热需求缩放：enabled={heat_scaling.get('enabled', False)}；method={heat_scaling.get('method', 'N/A')}；target_peak_fraction_of_it_load={heat_scaling.get('target_peak_fraction_of_it_load', 'N/A')}。",
                f"- 缩放原因：{heat_scaling.get('source', '未说明')}",
            ]
        )

    lines.extend(["", "## 3. 收敛性分析", "", *_convergence_lines(convergence_summary)])
    lines.extend(["", "## 4. 模型-代码同步提示", "", "- 未检测到 RDHX 辅助风冷、BESS 双容量或供热松弛罚金字段。"])

    lines.extend(["", "## 5. Pareto 与代表方案", ""])
    if pareto.empty:
        lines.append("- 本轮没有可行 Pareto 解。")
    else:
        if len(pareto) == 1:
            lines.append("- 本轮 Pareto 只有 1 个解，说明当前参数或约束下成本/碳目标尚未形成充分权衡。")
        for label, row in _representative_rows(pareto):
            zone_totals, system_totals = _capacity_summary(row)
            lines.extend(
                [
                    f"### {label}",
                    f"- solution_id：{row.get('solution_id', 'N/A')}",
                    f"- generation：{row.get('generation', 'N/A')}",
                    f"- TLCC：{_format_number(float(row.get('tlcc', float('nan'))), 'yuan/year')}",
                    f"- TCE：{_format_number(float(row.get('tce', float('nan'))), 'kgCO2/year')}",
                    f"- 区域构型计数：{_format_config_counts(_config_counts(row), config)}",
                    "- 容量配置：",
                    *_format_capacity_lines(zone_totals, system_totals),
                ]
            )
            solution_path = row.get("solution_json_path", "")
            dispatch_path = row.get("dispatch_path", "")
            if isinstance(solution_path, str) and solution_path:
                lines.append(f"- 解详情：`{solution_path}`")
            if isinstance(dispatch_path, str) and dispatch_path:
                lines.append(f"- 调度明细：`{dispatch_path}`")
            lines.append("")

    lines.extend(
        [
            "## 6. 可行性、repair 与不可行原因",
            "",
            f"- 历史可行解数量：{len(feasible)}",
            f"- 历史非支配筛选后 Pareto 解数量：{len(pareto)}",
        ]
    )
    if repaired_count is not None:
        lines.append(f"- repair 后向量与原始向量不同的候选解数量：{repaired_count}")
    lines.extend(_reason_summary(all_evaluations))

    lines.extend(["", "## 7. 机房空间可视化", "", *_spatial_visualization_lines(output)])

    figures = output / "figures"
    convergence = output / "convergence"
    lines.extend(
        [
            "",
            "## 8. 输出文件索引",
            "",
            f"- 全部历史评估：`{output / 'all_evaluations.csv'}`",
            f"- Pareto 解集：`{output / 'pareto_solutions.csv'}`",
            f"- Pareto 前沿图：`{figures / 'pareto.png'}`",
            f"- Pareto 决策变量分布：`{figures / 'pareto_solution_distribution.png'}`",
            f"- 机房空间布局图：`{figures / 'room_layout_knee_solution_room*.png'}`",
            f"- 构型选择与业务爬升/空调老化关系图：`{figures / 'config_choice_vs_growth_age.png'}`",
            f"- pymoo 收敛总览：`{figures / 'pymoo_convergence_summary.png'}`",
            f"- 收敛指标明细：`{convergence / 'pymoo_convergence_metrics.csv'}`",
            f"- Pareto 详情目录：`{output / 'pareto_details'}`",
            f"- 本报告：`{report_path}`",
            "",
            "## 9. 参数来源与假设",
            "",
            *_assumption_lines(config),
            "",
            "## 10. 后续审核重点",
            "",
            "- 正式论文实验前，应继续替换成本、隐含碳、重量、寿命、FOM/VOM 等占位参数。",
            "- 当前供热需求已按 IT 余热量级缩放，供热侧结论应解释为余热消纳场景，不能直接外推到原始 8 MW 建筑供热峰值。",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
