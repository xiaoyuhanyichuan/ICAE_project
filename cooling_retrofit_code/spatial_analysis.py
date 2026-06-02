from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def config_name_map(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("technology", {}).get("configurations", {})
    names: dict[int, str] = {}
    for key, value in raw.items():
        try:
            names[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return names


def select_knee_solution(pareto: pd.DataFrame) -> pd.Series:
    if pareto.empty:
        raise ValueError("Cannot select knee solution from an empty Pareto frame.")

    finite = pareto.copy()
    finite["tlcc_num"] = pd.to_numeric(finite["tlcc"], errors="coerce")
    finite["tce_num"] = pd.to_numeric(finite["tce"], errors="coerce")
    finite = finite[np.isfinite(finite["tlcc_num"]) & np.isfinite(finite["tce_num"])]
    if finite.empty:
        raise ValueError("Cannot select knee solution without finite TLCC/TCE values.")
    if len(finite) == 1:
        return finite.iloc[0]

    tlcc_span = finite["tlcc_num"].max() - finite["tlcc_num"].min()
    tce_span = finite["tce_num"].max() - finite["tce_num"].min()
    norm_cost = (
        pd.Series(0.0, index=finite.index)
        if tlcc_span <= 0
        else (finite["tlcc_num"] - finite["tlcc_num"].min()) / tlcc_span
    )
    norm_carbon = (
        pd.Series(0.0, index=finite.index)
        if tce_span <= 0
        else (finite["tce_num"] - finite["tce_num"].min()) / tce_span
    )
    knee_idx = (norm_cost.pow(2) + norm_carbon.pow(2)).idxmin()
    return finite.loc[knee_idx]


def _growth_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("scenario", {}).get("business_growth", {})


def _aging_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("scenario", {}).get("ac_aging", {})


def _rho_load_for_rack(rack_id: str, ac_unit: str, config: dict[str, Any]) -> float:
    growth = _growth_config(config)
    by_rack = growth.get("rho_load_by_rack", {})
    by_ac = growth.get("rho_load_by_ac_unit", {})
    default = _to_float(growth.get("default_rho_load", 0.0))
    if rack_id in by_rack:
        return _to_float(by_rack[rack_id], default)
    if ac_unit in by_ac:
        return _to_float(by_ac[ac_unit], default)
    return default


def _alpha_age_for_ac(ac_unit: str, config: dict[str, Any]) -> float:
    aging = _aging_config(config)
    by_ac = aging.get("alpha_age_by_ac_unit", {})
    default = _to_float(aging.get("default_alpha_age", 0.0))
    return _to_float(by_ac.get(ac_unit, default), default)


def _rack_label(rack_id: str) -> str:
    match = re.search(r"(\d+)$", str(rack_id))
    return str(int(match.group(1))) if match else str(rack_id)


def build_rack_layout_frame(
    knee_row: pd.Series,
    rack_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    required = {"rack_id", "row", "col", "ac_unit"}
    missing = sorted(required - set(rack_metadata.columns))
    if missing:
        raise ValueError(f"rack_metadata is missing required columns: {missing}")

    config_by_zone = _safe_json(knee_row.get("config_by_zone_json"))
    if not isinstance(config_by_zone, dict):
        raise ValueError("knee solution is missing config_by_zone_json.")

    names = config_name_map(config)
    frame = rack_metadata.copy()
    frame["solution_id"] = int(knee_row.get("solution_id", -1))
    frame["ac_unit"] = frame["ac_unit"].astype(str)
    frame["config_id"] = frame["ac_unit"].map(lambda zone: int(config_by_zone.get(zone, -1)))
    frame["config_name"] = frame["config_id"].map(lambda item: names.get(int(item), f"config_{int(item)}"))
    if "rho_load" not in frame.columns:
        frame["rho_load"] = frame.apply(
            lambda row: _rho_load_for_rack(str(row["rack_id"]), str(row["ac_unit"]), config),
            axis=1,
        )
    else:
        fallback_rho = frame.apply(
            lambda row: _rho_load_for_rack(str(row["rack_id"]), str(row["ac_unit"]), config),
            axis=1,
        )
        frame["rho_load"] = pd.to_numeric(frame["rho_load"], errors="coerce").fillna(fallback_rho)
    if "alpha_age" not in frame.columns:
        frame["alpha_age"] = frame["ac_unit"].map(lambda zone: _alpha_age_for_ac(str(zone), config))
    else:
        fallback_alpha = frame["ac_unit"].map(lambda zone: _alpha_age_for_ac(str(zone), config))
        frame["alpha_age"] = pd.to_numeric(frame["alpha_age"], errors="coerce").fillna(fallback_alpha)
    if "epsilon_age" not in frame.columns:
        frame["epsilon_age"] = frame["alpha_age"]
    else:
        frame["epsilon_age"] = pd.to_numeric(frame["epsilon_age"], errors="coerce").fillna(frame["alpha_age"])
    frame["rack_label"] = frame["rack_id"].map(_rack_label)
    if "hotspot_risk" not in frame.columns:
        frame["hotspot_risk"] = np.nan
    if "room_id" not in frame.columns:
        frame["room_id"] = "room01"
    frame["room_id"] = frame["room_id"].astype(str)

    col_values = pd.to_numeric(frame["col"], errors="coerce").fillna(0.0)
    row_values = pd.to_numeric(frame["row"], errors="coerce").fillna(0.0)
    frame["plot_col"] = col_values
    frame["plot_row"] = row_values

    return frame[
        [
            "solution_id",
            "rack_id",
            "rack_label",
            "room_id",
            "row",
            "col",
            "plot_row",
            "plot_col",
            "ac_unit",
            "config_id",
            "config_name",
            "rho_load",
            "alpha_age",
            "epsilon_age",
            "hotspot_risk",
        ]
    ]


def build_zone_relationship_frame(
    knee_row: pd.Series,
    rack_frame: pd.DataFrame,
    hourly: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    del config
    rows: list[dict[str, Any]] = []
    solution_id = int(knee_row.get("solution_id", -1))
    for ac_unit, group in rack_frame.groupby("ac_unit", sort=True):
        rack_cols = [
            f"{rack_id}_kw"
            for rack_id in group["rack_id"].astype(str)
            if f"{rack_id}_kw" in hourly.columns
        ]
        if rack_cols:
            zone_series = hourly[rack_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
            peak_it = float(zone_series.max())
        else:
            peak_it = 0.0

        first = group.iloc[0]
        rows.append(
            {
                "solution_id": solution_id,
                "ac_unit": ac_unit,
                "config_id": int(first["config_id"]),
                "config_name": str(first["config_name"]),
                "rack_count": int(len(group)),
                "rho_load_mean": float(pd.to_numeric(group["rho_load"], errors="coerce").mean()),
                "alpha_age": float(pd.to_numeric(group["alpha_age"], errors="coerce").mean()),
                "epsilon_age": float(pd.to_numeric(group["epsilon_age"], errors="coerce").mean()),
                "zone_peak_it_kw": peak_it,
                "zone_mean_hotspot_risk": float(pd.to_numeric(group["hotspot_risk"], errors="coerce").mean()),
                "zone_max_hotspot_risk": float(pd.to_numeric(group["hotspot_risk"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def _color_map(values: pd.Series) -> dict[int, tuple[float, float, float, float]]:
    import matplotlib.pyplot as plt

    unique = sorted({int(value) for value in values.dropna().tolist()})
    cmap = plt.get_cmap("tab10")
    return {value: cmap(index % 10) for index, value in enumerate(unique)}


def _room_layout_file(figures: Path, room_id: str) -> Path:
    safe_room_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(room_id)).strip("_") or "room"
    return figures / f"room_layout_knee_solution_{safe_room_id}.png"


def save_room_layout_plot(rack_frame: pd.DataFrame, knee_row: pd.Series, output_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    old_combined_path = figures / "room_layout_knee_solution.png"
    try:
        old_combined_path.unlink()
    except (FileNotFoundError, PermissionError):
        pass
    colors = _color_map(rack_frame["config_id"])
    rho = pd.to_numeric(rack_frame["rho_load"], errors="coerce").fillna(0.0)
    alpha = pd.to_numeric(rack_frame["alpha_age"], errors="coerce").fillna(0.0)
    rho_span = max(float(rho.max() - rho.min()), 1.0e-9)
    alpha_span = max(float(alpha.max() - alpha.min()), 1.0e-9)
    size_by_rack = 280.0 + 320.0 * (rho - float(rho.min())) / rho_span
    linewidth_by_rack = 1.0 + 3.0 * (alpha - float(alpha.min())) / alpha_span

    room_ids = sorted(rack_frame["room_id"].astype(str).unique().tolist())
    x_values = pd.to_numeric(rack_frame["plot_col"], errors="coerce")
    y_values = pd.to_numeric(rack_frame["plot_row"], errors="coerce")
    x_min, x_max = float(x_values.min()) - 1.0, float(x_values.max()) + 1.0
    y_min, y_max = float(y_values.min()) - 0.7, float(y_values.max()) + 0.7
    tlcc = _to_float(knee_row.get("tlcc"), float("nan"))
    tce = _to_float(knee_row.get("tce"), float("nan"))

    first_path: Path | None = None
    for room_id in room_ids:
        fig, ax = plt.subplots(figsize=(7.3, 5.0))
        legend_handles: dict[int, Any] = {}
        room_frame = rack_frame[rack_frame["room_id"].astype(str) == room_id]
        for config_id, group in room_frame.groupby("config_id", sort=True):
            edgecolors = [
                "red" if _to_float(value, 0.0) >= 0.8 else "black"
                for value in group["hotspot_risk"].tolist()
            ]
            scatter = ax.scatter(
                pd.to_numeric(group["plot_col"], errors="coerce"),
                pd.to_numeric(group["plot_row"], errors="coerce"),
                s=size_by_rack.loc[group.index],
                color=colors[int(config_id)],
                edgecolors=edgecolors,
                linewidths=linewidth_by_rack.loc[group.index],
                alpha=0.86,
                label=f"{int(config_id)} {group['config_name'].iloc[0]}",
            )
            legend_handles[int(config_id)] = scatter
            for _, row in group.iterrows():
                ax.text(
                    float(row["plot_col"]),
                    float(row["plot_row"]),
                    str(row["rack_label"]),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                )
                ax.text(
                    float(row["plot_col"]),
                    float(row["plot_row"]) + 0.24,
                    str(row["ac_unit"]),
                    ha="center",
                    va="bottom",
                    fontsize=4.6,
                    color="#333333",
                )
                ax.text(
                    float(row["plot_col"]),
                    float(row["plot_row"]) - 0.24,
                    f"{float(row['rho_load']):.2f}/{float(row['alpha_age']):.2f}",
                    ha="center",
                    va="top",
                    fontsize=4.4,
                    color="#222222",
                )
        ax.set_title(room_id, fontsize=12, fontweight="bold")
        ax.set_xlabel("Rack column")
        ax.set_ylabel("Rack row")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.invert_yaxis()
        ax.set_title(
            f"{room_id} retrofit configuration - knee solution {int(knee_row.get('solution_id', -1))}\n"
            f"TLCC={tlcc:,.0f} yuan/year, TCE={tce:,.0f} kgCO2/year; label=rho/alpha",
            fontsize=12,
        )
        handles = [legend_handles[key] for key in sorted(legend_handles)]
        labels = [handle.get_label() for handle in handles]
        if handles:
            ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Configuration")
        fig.tight_layout()
        path = _room_layout_file(figures, room_id)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        if first_path is None:
            first_path = path

    if first_path is None:
        path = _room_layout_file(figures, "empty")
        fig, ax = plt.subplots(figsize=(7.3, 5.0))
        ax.set_axis_off()
        ax.set_title("No rack layout data")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path
    return first_path


def save_growth_age_plot(zone_frame: pd.DataFrame, output_dir: str | Path) -> Path:
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "config_choice_vs_growth_age.png"
    colors = _color_map(zone_frame["config_id"])
    peak_values = pd.to_numeric(zone_frame["zone_peak_it_kw"], errors="coerce").fillna(0.0)
    max_peak = max(float(peak_values.max()), 1.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    for config_id, group in zone_frame.groupby("config_id", sort=True):
        sizes = 180.0 + 520.0 * pd.to_numeric(group["zone_peak_it_kw"], errors="coerce").fillna(0.0) / max_peak
        ax.scatter(
            pd.to_numeric(group["rho_load_mean"], errors="coerce"),
            pd.to_numeric(group["alpha_age"], errors="coerce"),
            s=sizes,
            color=colors[int(config_id)],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.82,
            label=f"{int(config_id)} {group['config_name'].iloc[0]}",
        )
        for _, row in group.iterrows():
            ax.text(
                float(row["rho_load_mean"]),
                float(row["alpha_age"]),
                str(row["ac_unit"]),
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.set_title("Configuration choice vs. business growth and AC aging")
    ax.set_xlabel("Zone mean business load growth intensity rho_load")
    ax.set_ylabel("Old AC capacity aging rate alpha_age")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", title="Configuration")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_spatial_analysis(
    pareto: pd.DataFrame,
    rack_metadata: pd.DataFrame,
    hourly: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if pareto.empty:
        return {"available": False, "reason": "empty_pareto"}
    try:
        knee = select_knee_solution(pareto)
        rack_frame = build_rack_layout_frame(knee, rack_metadata, config)
        zone_frame = build_zone_relationship_frame(knee, rack_frame, hourly, config)
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}

    spatial_dir = Path(output_dir) / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    rack_csv = spatial_dir / "knee_solution_rack_layout.csv"
    relationship_csv = spatial_dir / "config_choice_relationship.csv"
    rack_frame.to_csv(rack_csv, index=False, encoding="utf-8-sig")
    zone_frame.to_csv(relationship_csv, index=False, encoding="utf-8-sig")
    room_plot = save_room_layout_plot(rack_frame, knee, output_dir)
    relationship_plot = save_growth_age_plot(zone_frame, output_dir)

    return {
        "available": True,
        "solution_id": int(knee.get("solution_id", -1)),
        "room_layout_path": str(room_plot),
        "relationship_path": str(relationship_plot),
        "rack_layout_csv": str(rack_csv),
        "relationship_csv": str(relationship_csv),
        "zone_count": int(len(zone_frame)),
        "rack_count": int(len(rack_frame)),
    }
