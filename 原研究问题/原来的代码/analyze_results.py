from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_utils import ensure_time_series, load_config
from model import evaluate_solution
from result_utils import decode_vector


ROOT = Path(__file__).resolve().parents[1]


def fmt(v: float) -> str:
    if abs(v) >= 1e8:
        return f"{v/1e8:.3f}e8"
    if abs(v) >= 1e6:
        return f"{v/1e6:.3f}e6"
    return f"{v:.2f}"


def fmt_seconds(v: float) -> str:
    try:
        fv = float(v)
    except Exception:
        return "N/A"
    if not pd.notna(fv):
        return "N/A"
    return f"{fv:.2f} s"


PLAN_FIELDS = [
    ("n_lc", "液冷改造机柜数 n_lc", "count"),
    ("n_crac_remove", "拆除CRAC台数 n_crac_remove", "count"),
    ("cap_cdu_kw", "CDU容量 cap_cdu_kw", "kW"),
    ("cap_hp_kw", "热泵容量 cap_hp_kw", "kW"),
    ("cap_bess_e_kwh", "BESS能量容量 cap_bess_e_kwh", "kWh"),
    ("cap_bess_p_kw", "BESS功率容量 cap_bess_p_kw", "kW"),
    ("cap_tes_kwh", "TES容量 cap_tes_kwh", "kWh"),
    ("cap_tower_kw", "冷却塔扩容量 cap_tower_kw", "kW"),
    ("delta_cap_tower_kw", "冷却塔扩容量增量 delta_cap_tower_kw", "kW"),
    ("cap_tower_total_kw", "冷却塔总容量 cap_tower_total_kw", "kW"),
    ("cap_hx_kw", "板换容量 cap_hx_kw", "kW"),
]


def fmt_plan_value(v: float | bool, value_type: str) -> str:
    if value_type == "bool":
        return "是" if bool(v) else "否"
    if value_type == "count":
        return f"{float(v):.0f}"
    return f"{float(v):.2f}"


def _read_runtime_summary(results: Path) -> dict:
    path = results / "runtime_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_pymoo_convergence_summary(results: Path) -> dict:
    path = results / "convergence" / "pymoo_convergence_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    results = ROOT / "results"
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results / "pareto_solutions.csv")
    reps = pd.read_csv(results / "representative_solutions.csv", index_col=0)
    runtime = _read_runtime_summary(results)
    convergence = _read_pymoo_convergence_summary(results)
    cfg = load_config()
    ts = ensure_time_series(cfg)

    min_cost = reps.loc["min_cost"]
    min_carbon = reps.loc["min_carbon"]
    knee = reps.loc["knee"]

    tradeoff_cost = (min_carbon["tlcc_yuan"] - min_cost["tlcc_yuan"]) / max(min_cost["tlcc_yuan"], 1e-9)
    tradeoff_carbon = (min_cost["tce_kgco2"] - min_carbon["tce_kgco2"]) / max(min_cost["tce_kgco2"], 1e-9)

    lines = []
    lines.append("# 数据中心改造成本-碳排多目标优化结果分析")
    lines.append("")
    lines.append("## 1. 求解概览")
    lines.append(f"- 帕累托解数量: **{len(df)}**")
    lines.append(f"- 年化成本(TLCC)范围: **{fmt(df['tlcc_yuan'].min())} ~ {fmt(df['tlcc_yuan'].max())} 元**")
    lines.append(f"- 年化碳排(TCE)范围: **{fmt(df['tce_kgco2'].min())} ~ {fmt(df['tce_kgco2'].max())} kgCO2**")
    if runtime:
        lines.append(f"- 多目标优化总体运行时间: **{runtime.get('optimization_runtime_hms', 'N/A')}**（{fmt_seconds(runtime.get('optimization_runtime_s', float('nan')))}）")
        lines.append(
            f"- 求解配置: **{runtime.get('typical_day_scenarios', 'N/A')}** 个场景 / "
            f"外层 **{runtime.get('outer_engine', 'custom')}** / "
            f"并行 **{runtime.get('parallel_workers', 'N/A')}** worker / "
            f"内层 **{runtime.get('inner_threads', 'N/A')}** thread"
        )
        lines.append(
            f"- 内层建模模式: **{'template_reuse' if bool(runtime.get('persistent_template_enabled', False)) else 'fresh'}**"
        )
    lines.append("")
    if runtime:
        eval_stats = runtime.get("eval_time_stats_all_evals_s", {})
        build_stats = runtime.get("inner_build_time_stats_all_evals_s", {})
        opt_stats = runtime.get("inner_optimize_time_stats_all_evals_s", {})
        lines.append("## 2. 求解耗时统计")
        if eval_stats:
            lines.append(
                f"- 单个外层候选解的一次内层评估耗时范围: **{fmt_seconds(eval_stats.get('min', float('nan')))} ~ {fmt_seconds(eval_stats.get('max', float('nan')))}**"
            )
            lines.append(f"- 单次内层评估平均耗时: **{fmt_seconds(eval_stats.get('mean', float('nan')))}**")
        if build_stats:
            lines.append(
                f"- 其中模型构建耗时范围: **{fmt_seconds(build_stats.get('min', float('nan')))} ~ {fmt_seconds(build_stats.get('max', float('nan')))}**，平均 **{fmt_seconds(build_stats.get('mean', float('nan')))}**"
            )
        if opt_stats:
            lines.append(
                f"- 其中Gurobi求解耗时范围: **{fmt_seconds(opt_stats.get('min', float('nan')))} ~ {fmt_seconds(opt_stats.get('max', float('nan')))}**，平均 **{fmt_seconds(opt_stats.get('mean', float('nan')))}**"
            )
        if bool(runtime.get("persistent_template_enabled", False)):
            tpl_stats = runtime.get("template_update_time_stats_all_evals_s", {})
            lines.append(
                f"- 模板复用命中率: **{float(runtime.get('template_reuse_hit_rate_all_evals', float('nan'))) * 100:.2f}%**"
            )
            if tpl_stats:
                lines.append(
                    f"- 模板更新耗时范围: **{fmt_seconds(tpl_stats.get('min', float('nan')))} ~ {fmt_seconds(tpl_stats.get('max', float('nan')))}**，平均 **{fmt_seconds(tpl_stats.get('mean', float('nan')))}**"
                )
        if "diagnostic_relaxation_rate_all_evals" in runtime:
            lines.append(
                f"- 不可行诊断触发率: **{float(runtime.get('diagnostic_relaxation_rate_all_evals', 0.0)) * 100:.2f}%**，"
                f"诊断修复率: **{float(runtime.get('diagnostic_repair_rate_all_evals', 0.0)) * 100:.2f}%**，"
                f"修复后重求成功率: **{float(runtime.get('diagnostic_resolve_success_rate_all_evals', 0.0)) * 100:.2f}%**"
            )
            lines.append(
                f"- 不可行记忆命中率: **{float(runtime.get('infeasible_memory_hit_rate_all_evals', 0.0)) * 100:.2f}%**"
            )
        lines.append("")
    lines.append("## 3. pymoo 收敛性分析")
    outer_engine = str(runtime.get("outer_engine", cfg.get("solver", {}).get("outer_engine", "custom"))).lower()
    if outer_engine != "pymoo":
        lines.append("- 本轮外层算法不是 `pymoo`，未启用 pymoo Hypervolume / IGD+ 收敛性分析。")
    elif not convergence or not bool(convergence.get("available", False)):
        reason = convergence.get("reason", "missing_convergence_output") if convergence else "missing_convergence_output"
        lines.append(f"- 本轮使用 `pymoo`，但收敛性指标不可用，原因: `{reason}`。")
    else:
        rec_gen = convergence.get("recommended_generation", None)
        lines.append(f"- 最终 Hypervolume: **{float(convergence.get('final_hv', float('nan'))):.4f}**")
        lines.append(f"- 最终 IGD+: **{float(convergence.get('final_igd_plus', float('nan'))):.4f}**")
        lines.append(f"- 最终代可行个体数: **{int(convergence.get('final_n_feasible', 0))}**")
        lines.append(f"- 参考前沿口径: **{convergence.get('reference_front', 'history_nondominated')}**")
        if rec_gen is None:
            lines.append("- 未检测到连续边际收益低于阈值的稳定区间，建议暂不减少迭代次数。")
        else:
            lines.append(f"- 按当前阈值检测，建议迭代代数可优先试验调整到 **{int(rec_gen)}** 代附近。")
        lines.append("- 收敛指标文件: `results/convergence/pymoo_convergence_metrics.csv`")
        lines.append("- 代际种群快照: `results/convergence/pymoo_generation_population.csv`")
        lines.append("- 收敛图: `results/figures/pymoo_convergence_hv.png` / `pymoo_convergence_igd_plus.png` / `pymoo_convergence_summary.png`")
    lines.append("")

    lines.append("## 4. 代表性方案")
    for tag, row in reps.iterrows():
        lines.append(f"### {tag}")
        lines.append(f"- TLCC: {fmt(row['tlcc_yuan'])} 元")
        lines.append(f"- TCE: {fmt(row['tce_kgco2'])} kgCO2")
        lines.append("- 规划结果变量：")
        for col, label, value_type in PLAN_FIELDS:
            if col in row.index:
                suffix = ""
                if value_type == "kW":
                    suffix = " kW"
                elif value_type == "kWh":
                    suffix = " kWh"
                lines.append(f"  - {label}: {fmt_plan_value(row[col], value_type)}{suffix}")
        lines.append("")

    lines.append("## 5. 成本-碳权衡")
    lines.append(f"- 从最低成本方案切换到最低碳方案，成本变化: **{tradeoff_cost*100:.2f}%**")
    lines.append(f"- 对应可实现碳排下降: **{tradeoff_carbon*100:.2f}%**")
    lines.append("")
    lines.append("## 6. 结果解读")
    lines.append("- 低成本方案倾向于较低储能/回收设备投资，以减少CAPEX和固定运维。")
    lines.append("- 低碳方案倾向于提升液冷渗透率与余热利用能力，以减少电网购电碳排并增加热收益。")
    if "chiller_req_kw" in df.columns and "chiller_old_cap_kw" in df.columns:
        lines.append(
            f"- 冷机组约束校核: 需求峰值约 **{df['chiller_req_kw'].max():.2f} kW**，"
            f"小于现有容量 **{df['chiller_old_cap_kw'].iloc[0]:.2f} kW**。"
        )
    lines.append("- 折中(knee)方案通常在热泵+换热+适中储能上取得较好平衡，可作为工程优先候选。")
    lines.append("")
    lines.append("## 7. 注意事项")
    lines.append("- 原始文档中部分参数存在乱码，当前结果基于 `data/config.json` 中的默认补全值。")
    lines.append("- 若替换为你的精确参数与真实负荷序列，请重新运行求解并更新图表。")
    lines.append("- `task_balance_violation` 和 `slot_violation` 是旧输出兼容字段；当前机柜 IT 负荷已由 `rack_it_*_kw` 固定输入，正常求解时二者固定为 0。")

    lines.append("")
    lines.append("## 8. 运行调度分析（逐小时）")
    for tag, row in reps.iterrows():
        x = decode_vector(row)
        _, _, _, series = evaluate_solution(x, cfg, ts, return_series=True)
        s = pd.DataFrame(series)
        heat_supply = max(1e-9, float(s["q_heat_supply_kw"].sum()))
        hp_share = float(s["q_heat_hp_kw"].sum()) / heat_supply
        direct_share = float(s["q_heat_direct_kw"].sum()) / heat_supply
        tes_share = float(s["q_tes_dis_kw"].sum()) / heat_supply
        peak_grid = float(s["p_grid_kw"].max())
        peak_hour = int(s.loc[s["p_grid_kw"].idxmax(), "hour"])
        hp_hours = int((s["p_hp_kw"] > 1e-6).sum())
        tes_ch = float(s["q_tes_ch_kw"].sum())
        tes_dis = float(s["q_tes_dis_kw"].sum())
        bess_ch = float(s["p_bess_ch_kw"].sum())
        bess_dis = float(s["p_bess_dis_kw"].sum())

        lines.append(f"### {tag}")
        lines.append(f"- 峰值购电功率: **{peak_grid:.2f} kW**（发生在 hour={peak_hour}）")
        lines.append(f"- 供热结构占比: 直供 **{direct_share*100:.2f}%** / HP **{hp_share*100:.2f}%** / TES放热 **{tes_share*100:.2f}%**")
        lines.append(f"- HP运行小时数: **{hp_hours} h**")
        lines.append(f"- TES年充热/放热: **{tes_ch:.2f} / {tes_dis:.2f} kWh**")
        lines.append(f"- BESS年充电/放电: **{bess_ch:.2f} / {bess_dis:.2f} kWh**")
        lines.append(f"- 调度明细文件: `results/dispatch/{tag}_dispatch_timeseries.csv`")
        lines.append("")

    out = reports / "result_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("Saved report:", out)


if __name__ == "__main__":
    main()
