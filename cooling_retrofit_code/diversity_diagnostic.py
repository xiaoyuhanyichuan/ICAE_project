from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import load_config, load_input_data
from decision import DecisionSchema, PlanningDecision
from model import _capacity_cost
from run_optimization import infer_zone_ids


ZONE_CAPACITY_NAMES = ("cap_ac_new", "cap_rdhx", "cap_cdu")
SYSTEM_CAPACITY_NAMES = (
    "cap_ashp",
    "cap_wshp",
    "cap_bess",
    "cap_tes",
    "delta_cap_chiller",
    "delta_cap_tower",
)
CAPACITY_COLUMNS = (
    "sum_cap_ac_new_kw",
    "sum_cap_rdhx_kw",
    "sum_cap_cdu_kw",
    "cap_ashp_kw",
    "cap_wshp_kw",
    "cap_bess",
    "cap_tes",
    "delta_cap_chiller_kw",
    "delta_cap_tower_kw",
)
COMPONENT_COLUMNS = (
    "annualized_capex_yuan_per_year",
    "operating_cost_yuan_per_year",
    "annualized_embodied_kg_per_year",
    "operating_carbon_kg_per_year",
)
GROUP_COLUMNS = (
    "new_air_zone_count",
    "cold_plate_zone_count",
    "rdhx_zone_count",
    "ashp_zone_count",
    "wshp_zone_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose diversity differences between Pareto and dominated planning solutions."
    )
    parser.add_argument("--config", default="config.json", help="Path to project config.json.")
    parser.add_argument(
        "--evaluations",
        default=None,
        help="Path to all_evaluations.csv. Defaults to <output_dir>/all_evaluations.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostic tables, figures, and report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_root = _resolve_output_dir(config, config_path)
    evaluations_path = Path(args.evaluations).resolve() if args.evaluations else output_root / "all_evaluations.csv"
    diagnostic_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else output_root / "diagnostics" / "diversity"
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    evaluations = pd.read_csv(evaluations_path)
    data = load_input_data(config)
    zone_ids = infer_zone_ids(data)
    schema = DecisionSchema.from_config(zone_ids, config)

    enriched = enrich_evaluations(evaluations, schema, config)
    if enriched.empty:
        raise RuntimeError("No feasible finite evaluations were found for diversity diagnostics.")

    summary = build_group_summary(enriched)
    config_distribution = build_config_distribution(enriched, config)
    capacity_quantiles = build_capacity_quantiles(enriched)
    correlations = build_correlation_summary(enriched)

    enriched_path = diagnostic_dir / "diversity_enriched_evaluations.csv"
    summary_path = diagnostic_dir / "diversity_group_summary.csv"
    config_path_out = diagnostic_dir / "config_distribution.csv"
    quantiles_path = diagnostic_dir / "capacity_quantiles.csv"
    correlations_path = diagnostic_dir / "objective_capacity_correlations.csv"

    enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    config_distribution.to_csv(config_path_out, index=False, encoding="utf-8-sig")
    capacity_quantiles.to_csv(quantiles_path, index=False, encoding="utf-8-sig")
    correlations.to_csv(correlations_path, index=False, encoding="utf-8-sig")

    figure_paths = {
        "scatter": diagnostic_dir / "pareto_vs_dominated_scatter.png",
        "capacity": diagnostic_dir / "capacity_distribution.png",
        "components": diagnostic_dir / "objective_components.png",
        "configs": diagnostic_dir / "config_distribution.png",
    }
    plot_objective_scatter(enriched, figure_paths["scatter"])
    plot_capacity_distribution(enriched, figure_paths["capacity"])
    plot_objective_components(enriched, figure_paths["components"])
    plot_config_distribution(config_distribution, figure_paths["configs"])

    report = build_report(
        enriched=enriched,
        summary=summary,
        config_distribution=config_distribution,
        capacity_quantiles=capacity_quantiles,
        correlations=correlations,
        figure_paths=figure_paths,
        output_dir=diagnostic_dir,
    )
    report_path = diagnostic_dir / "diversity_diagnostic_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote diversity diagnostic report: {report_path}")


def enrich_evaluations(
    evaluations: pd.DataFrame,
    schema: DecisionSchema,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = evaluations.copy()
    frame["feasible_bool"] = frame["feasible"].map(_as_bool)
    for col in ("tlcc", "tce", "front"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    feasible = frame[
        frame["feasible_bool"]
        & np.isfinite(frame["tlcc"].astype(float))
        & np.isfinite(frame["tce"].astype(float))
    ].copy()
    if feasible.empty:
        return feasible

    if "front" not in feasible.columns or feasible["front"].isna().all():
        feasible["front"] = _compute_fronts(feasible[["tlcc", "tce"]].to_numpy(dtype=float))
    feasible["solution_class"] = np.where(feasible["front"].astype(float) == 0.0, "pareto", "dominated")

    records: list[dict[str, Any]] = []
    for idx, row in feasible.iterrows():
        try:
            decision = _decision_from_row(row, schema)
        except Exception as exc:
            records.append({"_row_index": idx, "decode_error": str(exc)})
            continue

        annualized_capex, annualized_embodied = _capacity_cost(decision, config)
        tlcc = float(row["tlcc"])
        tce = float(row["tce"])
        operating_cost = tlcc - annualized_capex
        operating_carbon = tce - annualized_embodied

        record = {
            "_row_index": idx,
            "decode_error": "",
            "config_signature": "-".join(str(decision.s_z[z]) for z in decision.zone_ids),
            "unique_config_count": len(set(decision.s_z.values())),
            "new_air_zone_count": int(sum(decision.n_z.values())),
            "cold_plate_zone_count": int(sum(decision.p_z.values())),
            "rdhx_zone_count": int(sum(decision.r_z.values())),
            "ashp_zone_count": int(sum(decision.a_z.values())),
            "wshp_zone_count": int(sum(decision.w_z.values())),
            "sum_cap_ac_new_kw": float(sum(decision.cap_ac_new.values())),
            "sum_cap_rdhx_kw": float(sum(decision.cap_rdhx.values())),
            "sum_cap_cdu_kw": float(sum(decision.cap_cdu.values())),
            "cap_ashp_kw": float(decision.cap_ashp),
            "cap_wshp_kw": float(decision.cap_wshp),
            "cap_bess": float(decision.cap_bess),
            "cap_tes": float(decision.cap_tes),
            "delta_cap_chiller_kw": float(decision.delta_cap_chiller),
            "delta_cap_tower_kw": float(decision.delta_cap_tower),
            "annualized_capex_yuan_per_year": float(annualized_capex),
            "annualized_embodied_kg_per_year": float(annualized_embodied),
            "operating_cost_yuan_per_year": float(operating_cost),
            "operating_carbon_kg_per_year": float(operating_carbon),
            "tlcc_without_weight_penalty": float(tlcc),
            "tce_without_weight_penalty": float(tce),
        }
        for config_id in range(7):
            record[f"config_{config_id}_zone_count"] = int(
                sum(1 for value in decision.s_z.values() if int(value) == config_id)
            )
        records.append(record)

    decoded = pd.DataFrame.from_records(records).set_index("_row_index")
    out = feasible.join(decoded, how="left")
    out = out[out["decode_error"].fillna("") == ""].copy()
    out.drop(columns=["feasible_bool", "decode_error"], inplace=True, errors="ignore")
    return out.reset_index(drop=True)


def build_group_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "tlcc",
        "tce",
        "soft_violation_kg",
        *CAPACITY_COLUMNS,
        *GROUP_COLUMNS,
        *COMPONENT_COLUMNS,
    )
    rows: list[dict[str, Any]] = []
    for solution_class, group in enriched.groupby("solution_class", sort=False):
        row: dict[str, Any] = {
            "solution_class": solution_class,
            "count": int(len(group)),
            "unique_config_signatures": int(group["config_signature"].nunique()),
        }
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_config_distribution(enriched: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    labels = config.get("technology", {}).get("configurations", {})
    records: list[dict[str, Any]] = []
    zone_count = max(1, _infer_zone_count(enriched))
    for solution_class, group in enriched.groupby("solution_class", sort=False):
        denominator = max(1, len(group) * zone_count)
        for config_id in range(7):
            count = int(group[f"config_{config_id}_zone_count"].sum())
            records.append(
                {
                    "solution_class": solution_class,
                    "config_id": config_id,
                    "config_label": str(labels.get(str(config_id), f"config_{config_id}")),
                    "zone_count": count,
                    "share": count / denominator,
                }
            )
    return pd.DataFrame.from_records(records)


def build_capacity_quantiles(enriched: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    for solution_class, group in enriched.groupby("solution_class", sort=False):
        for metric in CAPACITY_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            for q in quantiles:
                records.append(
                    {
                        "solution_class": solution_class,
                        "metric": metric,
                        "quantile": q,
                        "value": float(values.quantile(q)),
                    }
                )
    return pd.DataFrame.from_records(records)


def build_correlation_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    metrics = list(CAPACITY_COLUMNS) + ["soft_violation_kg", "operating_cost_yuan_per_year", "operating_carbon_kg_per_year"]
    records: list[dict[str, Any]] = []
    for metric in metrics:
        values = pd.to_numeric(enriched[metric], errors="coerce")
        for objective in ("tlcc", "tce"):
            obj = pd.to_numeric(enriched[objective], errors="coerce")
            corr = float(values.corr(obj)) if values.nunique(dropna=True) > 1 else float("nan")
            records.append({"metric": metric, "objective": objective, "pearson_corr": corr})
    return pd.DataFrame.from_records(records)


def plot_objective_scatter(enriched: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    dominated = enriched[enriched["solution_class"] == "dominated"]
    pareto = enriched[enriched["solution_class"] == "pareto"]
    ax.scatter(
        dominated["tlcc"] / 1e6,
        dominated["tce"] / 1e6,
        s=14,
        alpha=0.25,
        color="#9ca3af",
        label="Dominated",
        linewidths=0,
    )
    size = 40 + 80 * _normalize(pareto["soft_violation_kg"])
    ax.scatter(
        pareto["tlcc"] / 1e6,
        pareto["tce"] / 1e6,
        s=size,
        alpha=0.9,
        color="#d55e00",
        edgecolor="black",
        linewidths=0.4,
        label="Pareto",
    )
    ax.set_xlabel("TLCC (million yuan/year)")
    ax.set_ylabel("TCE (million kgCO2e/year)")
    ax.set_title("Pareto vs Dominated Objective Distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_capacity_distribution(enriched: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), dpi=160)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, CAPACITY_COLUMNS):
        data = [
            enriched.loc[enriched["solution_class"] == "pareto", metric].to_numpy(dtype=float),
            enriched.loc[enriched["solution_class"] == "dominated", metric].to_numpy(dtype=float),
        ]
        ax.boxplot(data, tick_labels=["Pareto", "Dominated"], showfliers=False, patch_artist=True)
        ax.set_title(_short_label(metric), fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes_flat[len(CAPACITY_COLUMNS) :]:
        ax.axis("off")
    fig.suptitle("Capacity Combination Distribution", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_objective_components(enriched: pd.DataFrame, path: Path) -> None:
    cost_cols = [
        "annualized_capex_yuan_per_year",
        "operating_cost_yuan_per_year",
    ]
    carbon_cols = [
        "annualized_embodied_kg_per_year",
        "operating_carbon_kg_per_year",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    _plot_component_bars(enriched, cost_cols, axes[0], "Cost Components", 1e6, "million yuan/year")
    _plot_component_bars(enriched, carbon_cols, axes[1], "Carbon Components", 1e6, "million kgCO2e/year")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_config_distribution(config_distribution: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    classes = ["pareto", "dominated"]
    x = np.arange(7)
    width = 0.36
    colors = {"pareto": "#d55e00", "dominated": "#6b7280"}
    for offset, solution_class in zip((-width / 2, width / 2), classes):
        sub = config_distribution[config_distribution["solution_class"] == solution_class].sort_values("config_id")
        ax.bar(
            x + offset,
            sub["share"].to_numpy(dtype=float),
            width=width,
            label=solution_class.capitalize(),
            color=colors[solution_class],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(7)])
    ax.set_xlabel("Configuration ID")
    ax.set_ylabel("Zone share")
    ax.set_title("Configuration Choice Distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    config_distribution: pd.DataFrame,
    capacity_quantiles: pd.DataFrame,
    correlations: pd.DataFrame,
    figure_paths: dict[str, Path],
    output_dir: Path,
) -> str:
    pareto = enriched[enriched["solution_class"] == "pareto"]
    dominated = enriched[enriched["solution_class"] == "dominated"]
    pareto_count = len(pareto)
    dominated_count = len(dominated)
    total = len(enriched)
    share = pareto_count / max(1, total)

    differences = _class_differences(enriched)
    config_diff = _config_share_differences(config_distribution)
    top_capacity_diffs = differences[differences["metric"].isin(CAPACITY_COLUMNS)].head(5)
    top_component_diffs = differences[differences["metric"].isin(COMPONENT_COLUMNS + ("soft_violation_kg",))].head(6)

    lines = [
        "# Pareto / Dominated ???????",
        "",
        "## 1. ????",
        "",
        f"- ?????: {total}",
        f"- Pareto ??: {pareto_count}, ?? {share:.2%}",
        f"- Dominated ??: {dominated_count}",
        f"- Pareto ??????: {pareto['config_signature'].nunique()}",
        f"- Dominated ??????: {dominated['config_signature'].nunique()}",
        "",
        "## 2. ????",
        "",
        *_diagnostic_findings(enriched, differences, config_diff),
        "",
        "## 3. ??????",
        "",
        _markdown_table(config_diff.head(10)),
        "",
        "## 4. ??????",
        "",
        _markdown_table(top_capacity_diffs),
        "",
        "## 5. ????????????????",
        "",
        _markdown_table(top_component_diffs),
        "",
        "## 6. ?????/??????",
        "",
        _markdown_table(correlations.reindex(correlations["pearson_corr"].abs().sort_values(ascending=False).index).head(12)),
        "",
        "## 7. ??",
        "",
    ]
    for label, path in figure_paths.items():
        rel = path.relative_to(output_dir)
        lines.append(f"- {label}: `{rel.as_posix()}`")
    lines.extend(
        [
            "",
            "## 8. ????",
            "",
            "- `diversity_enriched_evaluations.csv`: ???????????????????????",
            "- `diversity_group_summary.csv`: Pareto ? dominated ??????",
            "- `config_distribution.csv`: ???????",
            "- `capacity_quantiles.csv`: ??????",
            "- `objective_capacity_correlations.csv`: ?????/????????",
            "",
            "## 9. ????",
            "",
            "?? Pareto ? dominated ?????????????????/???????????????????????????????? Pareto ??????????????????? repair ???????????? Pareto ? dominated ????????????????????? repair ????????????",
        ]
    )
    return "\n".join(lines) + "\n"

def _decision_from_row(row: pd.Series, schema: DecisionSchema) -> PlanningDecision:
    vector_text = str(row.get("repaired_vector_json", "") or "")
    if vector_text.strip():
        vector = np.asarray(json.loads(vector_text), dtype=float)
        return schema.decode(vector)
    raw_text = str(row.get("raw_vector_json", "") or "")
    if raw_text.strip():
        vector = np.asarray(json.loads(raw_text), dtype=float)
        return schema.decode(vector)
    raise ValueError("Missing repaired_vector_json and raw_vector_json.")


def _compute_fronts(objectives: np.ndarray) -> np.ndarray:
    n = objectives.shape[0]
    fronts = np.full(n, -1, dtype=int)
    dominated_counts = np.zeros(n, dtype=int)
    dominates: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominates[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                dominated_counts[i] += 1
        if dominated_counts[i] == 0:
            fronts[i] = 0
            first_front.append(i)
    current = first_front
    rank = 0
    while current:
        next_front: list[int] = []
        for i in current:
            for j in dominates[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    fronts[j] = rank + 1
                    next_front.append(j)
        rank += 1
        current = next_front
    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _resolve_output_dir(config: dict[str, Any], config_path: Path) -> Path:
    output_dir = Path(config.get("paths", {}).get("output_dir", "results"))
    if output_dir.is_absolute():
        return output_dir
    return config_path.parent / output_dir


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _infer_zone_count(enriched: pd.DataFrame) -> int:
    cols = [f"config_{idx}_zone_count" for idx in range(7)]
    if not all(col in enriched.columns for col in cols):
        return 1
    counts = enriched[cols].sum(axis=1)
    return int(max(1, counts.max()))


def _normalize(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    if arr.size == 0:
        return arr
    min_value = float(np.nanmin(arr))
    max_value = float(np.nanmax(arr))
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value == min_value:
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def _plot_component_bars(
    enriched: pd.DataFrame,
    columns: list[str],
    ax: plt.Axes,
    title: str,
    scale: float,
    ylabel: str,
) -> None:
    classes = ["pareto", "dominated"]
    labels = [_short_label(col) for col in columns]
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes), dtype=float)
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    for col, label, color in zip(columns, labels, colors):
        means = [
            float(enriched.loc[enriched["solution_class"] == cls, col].mean()) / scale
            for cls in classes
        ]
        ax.bar(x, means, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += np.asarray(means)
    ax.set_xticks(x)
    ax.set_xticklabels(["Pareto", "Dominated"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)


def _class_differences(enriched: pd.DataFrame) -> pd.DataFrame:
    pareto = enriched[enriched["solution_class"] == "pareto"]
    dominated = enriched[enriched["solution_class"] == "dominated"]
    metrics = (
        "tlcc",
        "tce",
        "soft_violation_kg",
        *CAPACITY_COLUMNS,
        *GROUP_COLUMNS,
        *COMPONENT_COLUMNS,
    )
    records: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in enriched.columns:
            continue
        p = pd.to_numeric(pareto[metric], errors="coerce")
        d = pd.to_numeric(dominated[metric], errors="coerce")
        p_mean = float(p.mean())
        d_mean = float(d.mean())
        pooled = float(np.sqrt((p.var(ddof=0) + d.var(ddof=0)) / 2.0))
        records.append(
            {
                "metric": metric,
                "pareto_mean": p_mean,
                "dominated_mean": d_mean,
                "pareto_minus_dominated": p_mean - d_mean,
                "relative_diff": (p_mean - d_mean) / abs(d_mean) if d_mean else np.nan,
                "standardized_diff": (p_mean - d_mean) / pooled if pooled else np.nan,
            }
        )
    out = pd.DataFrame.from_records(records)
    return out.reindex(out["standardized_diff"].abs().sort_values(ascending=False).index)


def _config_share_differences(config_distribution: pd.DataFrame) -> pd.DataFrame:
    pivot = config_distribution.pivot_table(
        index=["config_id", "config_label"],
        columns="solution_class",
        values="share",
        aggfunc="sum",
    ).reset_index()
    for col in ("pareto", "dominated"):
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["pareto_minus_dominated_share"] = pivot["pareto"] - pivot["dominated"]
    return pivot.sort_values("pareto_minus_dominated_share", key=lambda s: s.abs(), ascending=False)


def _diagnostic_findings(
    enriched: pd.DataFrame,
    differences: pd.DataFrame,
    config_diff: pd.DataFrame,
) -> list[str]:
    pareto = enriched[enriched["solution_class"] == "pareto"]
    dominated = enriched[enriched["solution_class"] == "dominated"]
    findings: list[str] = []
    config_line = config_diff.iloc[0]
    findings.append(
        "- 构型差异最大的是 "
        f"`{int(config_line['config_id'])}:{config_line['config_label']}`，"
        f"Pareto 份额比 dominated 高/低 {float(config_line['pareto_minus_dominated_share']):+.2%}。"
    )
    for metric in ("soft_violation_kg", "operating_cost_yuan_per_year", "operating_carbon_kg_per_year"):
        row = differences[differences["metric"] == metric]
        if row.empty:
            continue
        item = row.iloc[0]
        findings.append(
            f"- `{metric}` 的 Pareto 均值为 {_fmt(float(item['pareto_mean']))}，"
            f"dominated 均值为 {_fmt(float(item['dominated_mean']))}，"
            f"差异为 {_fmt(float(item['pareto_minus_dominated']))}。"
        )
    if pareto["config_signature"].nunique() <= max(2, len(pareto) // 3):
        findings.append("- Pareto 解的构型签名数量偏少，说明非支配前沿上的有效拓扑多样性较集中。")
    if dominated["operating_carbon_kg_per_year"].std(ddof=0) > 0:
        carbon_gap = abs(
            pareto["operating_carbon_kg_per_year"].mean()
            - dominated["operating_carbon_kg_per_year"].mean()
        )
        if carbon_gap < 0.05 * dominated["operating_carbon_kg_per_year"].std(ddof=0):
            findings.append("- Pareto 与 dominated 的运行碳均值差距很小，低碳差异可能主要来自容量/构型和隐含碳，而不是运行调度。")
    return findings


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_无数据_"
    sub = frame.head(max_rows).copy()
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: _fmt(float(x)))
    headers = [str(col) for col in sub.columns]
    rows = [[str(value) for value in row] for row in sub.to_numpy(dtype=object)]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def _short_label(name: str) -> str:
    replacements = {
        "sum_cap_ac_new_kw": "AC new",
        "sum_cap_rdhx_kw": "RDHX",
        "sum_cap_cdu_kw": "CDU",
        "cap_ashp_kw": "ASHP",
        "cap_wshp_kw": "WSHP",
        "cap_bess": "BESS",
        "cap_tes": "TES",
        "delta_cap_chiller_kw": "Chiller add",
        "delta_cap_tower_kw": "Tower add",
        "annualized_capex_yuan_per_year": "Capex",
        "operating_cost_yuan_per_year": "Op cost",
        "annualized_embodied_kg_per_year": "Embodied",
        "operating_carbon_kg_per_year": "Op carbon",
    }
    return replacements.get(name, name.replace("_", " "))


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 1e7:
        return f"{value:.3e}"
    if abs_value >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4g}"


if __name__ == "__main__":
    main()
