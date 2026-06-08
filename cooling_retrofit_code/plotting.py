from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def save_pareto_plot(pareto: pd.DataFrame, output_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "pareto.png"

    fig, ax = plt.subplots()
    ax.scatter(pareto["tlcc"], pareto["tce"])
    ax.set_xlabel("TLCC (yuan/year)")
    ax.set_ylabel("TCE (kgCO2/year)")
    ax.set_title("Pareto Front")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def save_convergence_plots(metrics: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty or not {"generation", "hv", "igd_plus"}.issubset(metrics.columns):
        return {}

    frame = metrics.copy()
    frame["generation"] = pd.to_numeric(frame["generation"], errors="coerce")
    frame["hv"] = pd.to_numeric(frame["hv"], errors="coerce")
    frame["igd_plus"] = pd.to_numeric(frame["igd_plus"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["generation", "hv", "igd_plus"])
    if frame.empty:
        return {}

    summary_path = figures_dir / "pymoo_convergence_summary.png"
    hv_path = figures_dir / "pymoo_convergence_hv.png"
    igd_path = figures_dir / "pymoo_convergence_igd_plus.png"

    fig, ax_hv = plt.subplots(figsize=(10, 5.6))
    ax_igd = ax_hv.twinx()
    hv_line = ax_hv.plot(
        frame["generation"],
        frame["hv"],
        color="#1f77b4",
        marker="o",
        linewidth=2.0,
        label="HV",
    )
    igd_line = ax_igd.plot(
        frame["generation"],
        frame["igd_plus"],
        color="#d95f02",
        marker="s",
        linewidth=2.0,
        label="IGD+",
    )
    ax_hv.set_xlabel("Generation")
    ax_hv.set_ylabel("Hypervolume", color="#1f77b4")
    ax_igd.set_ylabel("IGD+", color="#d95f02")
    ax_hv.tick_params(axis="y", labelcolor="#1f77b4")
    ax_igd.tick_params(axis="y", labelcolor="#d95f02")
    ax_hv.grid(alpha=0.25)
    lines = hv_line + igd_line
    ax_hv.legend(lines, [line.get_label() for line in lines], loc="best")
    ax_hv.set_title("pymoo Convergence Summary")
    fig.tight_layout()
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(frame["generation"], frame["hv"], marker="o", color="#1f77b4")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume by Generation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(hv_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(frame["generation"], frame["igd_plus"], marker="s", color="#d95f02")
    ax.set_xlabel("Generation")
    ax.set_ylabel("IGD+")
    ax.set_title("IGD+ by Generation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(igd_path, dpi=180)
    plt.close(fig)

    return {
        "summary": summary_path,
        "hv": hv_path,
        "igd_plus": igd_path,
    }


def save_pareto_distribution_plot(pareto: pd.DataFrame, output_dir: str | Path, config: dict | None = None) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    values = pareto_distribution_frame(pareto, config or {})
    if values.empty:
        return None

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "pareto_solution_distribution.png"

    columns = list(values.columns)
    n_cols = 3
    n_rows = int(math.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.0, 3.7 * n_rows))
    axes_array = np.asarray(axes).reshape(-1)

    for index, column in enumerate(columns):
        ax = axes_array[index]
        series = pd.to_numeric(values[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            ax.axis("off")
            continue
        bins = min(20, max(3, int(np.sqrt(len(series))) + 1))
        ax.hist(series, bins=bins, alpha=0.75, color="#1f77b4")
        ax.set_title(column)
        ax.grid(alpha=0.20)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)

    for index in range(len(columns), len(axes_array)):
        axes_array[index].axis("off")

    fig.suptitle("Decision Variable Distribution in Pareto Set", y=0.995)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def save_benchmark_comparison_plot(comparison: pd.DataFrame, output_dir: str | Path) -> Path | None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    required = {"scenario_key", "scenario_label", "tlcc", "tce"}
    if comparison.empty or not required.issubset(comparison.columns):
        return None

    frame = comparison.copy()
    panel_specs = [
        ("tlcc", "Annualized Cost", "TLCC (yuan/year)", "#1f77b4", None),
        ("tce", "Annualized Carbon", "TCE (kgCO2/year)", "#d95f02", None),
        ("pue", "Power Usage Effectiveness", "PUE", "#2ca02c", None),
        ("operational_cost_yuan", "Operating Cost", "yuan/year", "#9467bd", None),
        ("operational_carbon_kg", "Operating Carbon", "kgCO2/year", "#8c564b", None),
        ("waste_heat_recovery_rate", "Waste Heat Recovery", "recovery rate", "#17becf", "percent"),
    ]
    for column, *_ in panel_specs:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["tlcc", "tce"])
    if frame.empty:
        return None
    available_specs = [
        spec for spec in panel_specs if spec[0] in frame.columns and frame[spec[0]].notna().any()
    ]
    if len(available_specs) < 2:
        return None

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "benchmark_comparison.png"

    labels = frame["scenario_label"].astype(str).tolist()
    x = np.arange(len(frame))
    panel_count = len(available_specs)
    ncols = min(3, panel_count)
    nrows = int(np.ceil(panel_count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (column, title, ylabel, color, style) in zip(axes, available_specs):
        ax.bar(x, frame[column], color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if style == "percent":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.22)
    for ax in axes[panel_count:]:
        ax.axis("off")
    fig.suptitle("Baseline vs Refined Retrofit")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def pareto_distribution_frame(pareto: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    configurations = config.get("technology", {}).get("configurations", {})
    for _, row in pareto.iterrows():
        output: dict[str, float] = {}
        configs = _safe_json(row.get("config_by_zone_json"))
        if isinstance(configs, dict):
            for config_id, config_name in configurations.items():
                count = sum(1 for value in configs.values() if _config_key(value) == str(config_id))
                if count:
                    output[f"n_{config_name}"] = float(count)
            if configurations:
                for config_id, config_name in configurations.items():
                    output.setdefault(f"n_{config_name}", 0.0)
            else:
                for value in configs.values():
                    key = _config_key(value)
                    output[f"n_config_{key}"] = output.get(f"n_config_{key}", 0.0) + 1.0

        zone_caps = _safe_json(row.get("zone_capacity_json"))
        if isinstance(zone_caps, dict):
            for key, by_zone in zone_caps.items():
                if isinstance(by_zone, dict):
                    output[_capacity_label(key)] = _sum_numeric(by_zone.values())

        system_caps = _safe_json(row.get("system_capacity_json"))
        if isinstance(system_caps, dict):
            for key, value in system_caps.items():
                number = _to_float(value)
                if number is not None:
                    output[_capacity_label(key)] = number

        if output:
            rows.append(output)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows).fillna(0.0)
    preferred = [
        "n_existing_ac",
        "n_new_ac",
        "n_new_ac_with_ashp",
        "n_new_ac_high_efficiency",
        "n_cold_plate_cdu_wshp",
        "n_rdhx_wshp",
        "n_cold_plate_cdu_wshp_plus_air",
        "cap_ac_new_kw",
        "cap_rdhx_kw",
        "cap_cdu_kw",
        "cap_ashp_kw",
        "cap_wshp_kw",
        "cap_bess_kwh",
        "cap_tes_kwh",
        "delta_cap_chiller_kw",
        "delta_cap_tower_kw",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered]


def _safe_json(value) -> object | None:
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


def _is_integer_like(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer()


def _config_key(value) -> str:
    if _is_integer_like(value):
        return str(int(float(value)))
    return str(value)


def _sum_numeric(values) -> float:
    total = 0.0
    for value in values:
        number = _to_float(value)
        if number is not None:
            total += number
    return total


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _capacity_label(key: str) -> str:
    labels = {
        "cap_ac_new": "cap_ac_new_kw",
        "cap_rdhx": "cap_rdhx_kw",
        "cap_cdu": "cap_cdu_kw",
        "cap_ashp": "cap_ashp_kw",
        "cap_wshp": "cap_wshp_kw",
        "cap_bess": "cap_bess_kwh",
        "cap_tes": "cap_tes_kwh",
        "delta_cap_chiller": "delta_cap_chiller_kw",
        "delta_cap_tower": "delta_cap_tower_kw",
    }
    return labels.get(str(key), str(key))
