from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from data_utils import ensure_time_series, load_config
from model import build_typical_day_data, evaluate_solution
from result_utils import decode_vector


ROOT = Path(__file__).resolve().parents[1]

POWER_STYLES = {
    "p_grid_kw": {"label": "Grid", "color": "#1f77b4", "linewidth": 2.2, "linestyle": "-"},
    "it_total_kw": {"label": "IT", "color": "#111111", "linewidth": 1.8, "linestyle": "--"},
    "p_hp_kw": {"label": "HP", "color": "#ff7f0e", "linewidth": 1.6, "linestyle": "-"},
    "p_cdu_kw": {"label": "CDU", "color": "#2ca02c", "linewidth": 1.6, "linestyle": "-"},
    "p_crac_kw": {"label": "CRAC", "color": "#17becf", "linewidth": 1.6, "linestyle": "-."},
    "p_chiller_kw": {"label": "Chiller", "color": "#9467bd", "linewidth": 1.6, "linestyle": "-"},
    "p_tower_kw": {"label": "Tower", "color": "#8c564b", "linewidth": 1.6, "linestyle": ":"},
    "p_cooling_total_kw": {"label": "Cooling total", "color": "#7f7f7f", "linewidth": 2.0, "linestyle": "--"},
}

HEAT_STYLES = {
    "heat_demand_kw": {"label": "Heat demand", "color": "#111111", "linewidth": 2.0, "linestyle": "--"},
    "q_heat_supply_kw": {"label": "Heat supply", "color": "#d62728", "linewidth": 2.2, "linestyle": "-"},
    "q_heat_direct_kw": {"label": "Direct HX", "color": "#ff9896", "linewidth": 1.6, "linestyle": "-"},
    "q_heat_hp_kw": {"label": "HP heat", "color": "#ff7f0e", "linewidth": 1.6, "linestyle": "-"},
    "q_tes_dis_kw": {"label": "TES discharge", "color": "#2ca02c", "linewidth": 1.6, "linestyle": "-"},
    "q_tes_ch_kw": {"label": "TES charge", "color": "#1f77b4", "linewidth": 1.8, "linestyle": ":"},
}

PYMOO_PCP_COLS = [
    "tlcc_yuan",
    "tce_kgco2",
    "n_lc",
    "n_crac_remove",
    "cap_cdu_kw",
    "cap_hp_kw",
    "cap_bess_e_kwh",
    "cap_tes_kwh",
    "cap_hx_kw",
]

PYMOO_PCP_LABELS = {
    "tlcc_yuan": "TLCC",
    "tce_kgco2": "TCE",
    "n_lc": "LC racks",
    "n_crac_remove": "CRAC rm",
    "cap_cdu_kw": "CDU",
    "cap_hp_kw": "HP",
    "cap_bess_e_kwh": "BESS",
    "cap_tes_kwh": "TES",
    "cap_hx_kw": "HX",
}


def _disable_x_offset(ax) -> None:
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)


def _finite_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        return pd.DataFrame(index=df.index)

    numeric = df[available_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    return numeric.loc[finite_mask]


def _plot_pymoo_pareto_front(df: pd.DataFrame, reps: pd.DataFrame, fig_dir: Path) -> None:
    try:
        from pymoo.visualization.scatter import Scatter
    except ImportError:
        print("pymoo is not installed; skipped pymoo Pareto visualization.")
        return

    obj_cols = ["tlcc_yuan", "tce_kgco2"]
    obj = _finite_numeric_frame(df, obj_cols)
    if obj.empty:
        return

    scale = np.array([1e6, 1e6], dtype=float)
    pareto_f = obj[obj_cols].to_numpy(dtype=float) / scale

    plot = Scatter(
        title="pymoo Pareto Front: Cost vs Carbon",
        labels=["Annualized TLCC (million yuan)", "Annualized TCE (million kgCO2)"],
        figsize=(8, 6),
    )
    plot.add(pareto_f, color="#1f77b4", s=28, alpha=0.72, label="Pareto solutions")

    rep_obj = _finite_numeric_frame(reps, obj_cols)
    if not rep_obj.empty:
        rep_f = rep_obj[obj_cols].to_numpy(dtype=float) / scale
        plot.add(rep_f, color="#d95f02", marker="x", s=90, label="Representative solutions")

    plot.save(str(fig_dir / "pareto_front_pymoo.png"))


def _plot_pymoo_solution_distribution(df: pd.DataFrame, reps: pd.DataFrame, fig_dir: Path) -> None:
    try:
        from pymoo.visualization.pcp import PCP
    except ImportError:
        print("pymoo is not installed; skipped pymoo solution-distribution visualization.")
        return

    cols = [c for c in PYMOO_PCP_COLS if c in df.columns]
    values = _finite_numeric_frame(df, cols)
    if values.empty or len(values.columns) < 2:
        return

    labels = [PYMOO_PCP_LABELS.get(c, c) for c in values.columns]
    pareto_x = values.to_numpy(dtype=float)

    plot = PCP(
        title="pymoo Pareto Solution Distribution",
        labels=labels,
        normalize_each_axis=True,
        figsize=(11, 5.8),
        n_ticks=4,
    )
    plot.add(pareto_x, color="#2b8cbe", alpha=0.22, linewidth=1.0)

    rep_values = _finite_numeric_frame(reps, list(values.columns))
    if not rep_values.empty:
        plot.add(rep_values.to_numpy(dtype=float), color="#e66101", alpha=0.95, linewidth=2.4)

    plot.save(str(fig_dir / "solution_distribution_pymoo.png"))


def _plot_series(ax, df: pd.DataFrame, styles: dict[str, dict[str, object]], keys: list[str]) -> None:
    for key in keys:
        st = styles[key]
        ax.plot(
            df["hour"],
            df[key],
            label=str(st["label"]),
            color=str(st["color"]),
            linewidth=float(st["linewidth"]),
            linestyle=str(st["linestyle"]),
        )


def _plot_dispatch_by_scenario(
    df: pd.DataFrame,
    scenario_names: list[str],
    styles: dict[str, dict[str, object]],
    keys: list[str],
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    n_scenarios = len(scenario_names)
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(13.0, 2.8 * n_scenarios), sharex=True)
    if n_scenarios == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    for sid, (ax, scenario_name) in enumerate(zip(axes, scenario_names)):
        s_part = df.loc[df["scenario"] == sid].sort_values("hour")
        _plot_series(ax, s_part, styles, keys)
        if sid == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.set_ylabel(y_label)
        ax.set_title(str(scenario_name), loc="left", fontsize=10, pad=6)
        ax.set_xlim(0, 23)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Hour within representative day")
    fig.suptitle(title, y=0.995)
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=min(4, len(legend_labels)),
        fontsize=8,
        frameon=True,
        loc="upper center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    results = ROOT / "results"
    fig_dir = results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dispatch_dir = results / "dispatch"
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results / "pareto_solutions.csv")
    reps = pd.read_csv(results / "representative_solutions.csv", index_col=0)
    cfg = load_config()
    ts = ensure_time_series(cfg)
    td = build_typical_day_data(cfg, ts, count_build=False)

    _plot_pymoo_pareto_front(df, reps, fig_dir)
    _plot_pymoo_solution_distribution(df, reps, fig_dir)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["tlcc_yuan"] / 1e6, df["tce_kgco2"] / 1e6, s=25, alpha=0.7, label="Pareto solutions")
    for name, row in reps.iterrows():
        plt.scatter(row["tlcc_yuan"] / 1e6, row["tce_kgco2"] / 1e6, s=80, marker="x", label=name)
    plt.xlabel("Annualized TLCC (million yuan)")
    plt.ylabel("Annualized TCE (million kgCO2)")
    plt.title("Pareto Front: Cost vs Carbon")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pareto_front.png", dpi=180)
    plt.close()

    vars_to_plot = ["n_lc", "cap_hp_kw", "cap_bess_e_kwh", "cap_tes_kwh", "cap_cdu_kw", "cap_tower_kw", "cap_hx_kw"]
    axs_n = len(vars_to_plot)
    cols = 3
    rows = (axs_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = axes.flatten()
    for i, c in enumerate(vars_to_plot):
        axes[i].hist(df[c], bins=20, alpha=0.75)
        axes[i].set_title(c)
        _disable_x_offset(axes[i])
        axes[i].grid(alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Decision Variable Distribution in Pareto Set", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "decision_distribution.png", dpi=180)
    plt.close()

    comp_cols = ["tlcc_yuan", "tce_kgco2", "n_lc", "cap_hp_kw", "cap_bess_e_kwh", "cap_tes_kwh"]
    comp = reps[comp_cols]
    comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + 1e-9)

    plt.figure(figsize=(9, 5))
    x = range(len(comp_cols))
    for idx, (name, row) in enumerate(comp_norm.iterrows()):
        plt.plot(x, row.values, marker="o", label=name)
    plt.xticks(list(x), comp_cols, rotation=25, ha="right")
    plt.ylabel("Normalized value")
    plt.title("Representative Solutions Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "representative_comparison.png", dpi=180)
    plt.close()

    # Dispatch visualization and hourly export for representative solutions
    for tag, row in reps.iterrows():
        x = decode_vector(row)
        _, _, _, series = evaluate_solution(x, cfg, ts, return_series=True, td=td)
        s = pd.DataFrame(series)
        s.to_csv(dispatch_dir / f"{tag}_dispatch_timeseries.csv", index=False, encoding="utf-8-sig")
        s["p_cooling_total_kw"] = s["p_cdu_kw"] + s["p_crac_kw"] + s["p_chiller_kw"] + s["p_tower_kw"]

        _plot_dispatch_by_scenario(
            s,
            td.scenario_names,
            POWER_STYLES,
            [
                "p_grid_kw",
                "it_total_kw",
                "p_cooling_total_kw",
                "p_hp_kw",
                "p_cdu_kw",
                "p_crac_kw",
                "p_chiller_kw",
                "p_tower_kw",
            ],
            "Power (kW)",
            f"Dispatch Power Profile by Representative Day - {tag}",
            fig_dir / f"dispatch_power_{tag}.png",
        )

        _plot_dispatch_by_scenario(
            s,
            td.scenario_names,
            HEAT_STYLES,
            [
                "heat_demand_kw",
                "q_heat_supply_kw",
                "q_heat_direct_kw",
                "q_heat_hp_kw",
                "q_tes_dis_kw",
                "q_tes_ch_kw",
            ],
            "Heat flow (kW)",
            f"Dispatch Heat Profile by Representative Day - {tag}",
            fig_dir / f"dispatch_heat_{tag}.png",
        )

    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
