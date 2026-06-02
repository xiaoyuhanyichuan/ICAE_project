from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _get_total_racks(config: Dict[str, Any]) -> int:
    ex = config.get("existing", {})
    return int(ex["n_racks_total"])


def get_rack_load_cols(config: Dict[str, Any]) -> List[str]:
    rack_cfg = config.get("model2", {}).get("rack_load", {})
    prefix = str(rack_cfg.get("rack_col_prefix", "rack_it_"))
    width = int(rack_cfg.get("rack_col_width", 3))
    return [f"{prefix}{i:0{width}d}_kw" for i in range(_get_total_racks(config))]


def load_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or (ROOT / "data" / "config.json")
    with cfg_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _synthetic_total_it(config: Dict[str, Any], n: int) -> np.ndarray:
    rng = np.random.default_rng(config["project"]["random_seed"])
    hour = np.arange(n)
    hod = hour % 24
    doy = (hour // 24) % 365
    base = 9600.0 + 1800.0 * np.sin(2 * np.pi * hod / 24.0 - 0.8) + 900.0 * np.sin(2 * np.pi * doy / 365.0)
    return np.clip(base + rng.normal(0, 360.0, size=n), 6000.0, 14000.0)


def _infer_total_it_kw(config: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    if "it_total_kw" in df.columns:
        return df["it_total_kw"].to_numpy(dtype=float)
    return _synthetic_total_it(config, len(df))


def _synthetic_rack_loads(config: Dict[str, Any], total_it_kw: np.ndarray) -> np.ndarray:
    rack_cfg = config.get("model2", {}).get("rack_load", {})
    n_r = _get_total_racks(config)
    n = int(len(total_it_kw))
    rng = np.random.default_rng(int(config.get("model2", {}).get("seed", config["project"]["random_seed"] + 1000)) + 17)

    hetero_cv = max(0.0, float(rack_cfg.get("synthetic_heterogeneity_cv", 0.25)))
    noise_cv = max(0.0, float(rack_cfg.get("synthetic_noise_cv", 0.05)))
    phase_std = max(0.0, float(rack_cfg.get("synthetic_phase_std_h", 1.5)))

    hour = np.arange(n, dtype=float)
    hod = hour % 24.0
    doy = (hour // 24.0) % 365.0
    weights = rng.lognormal(mean=0.0, sigma=max(1e-6, hetero_cv), size=n_r)
    weights = weights / np.sum(weights)
    phase = rng.normal(0.0, phase_std, size=n_r)
    daily_amp = rng.uniform(0.02, 0.10, size=n_r)
    seasonal_amp = rng.uniform(0.00, 0.05, size=n_r)

    raw = np.zeros((n_r, n), dtype=float)
    for i in range(n_r):
        daily = 1.0 + daily_amp[i] * np.sin(2.0 * np.pi * ((hod - phase[i]) / 24.0))
        seasonal = 1.0 + seasonal_amp[i] * np.sin(2.0 * np.pi * ((doy - 170.0 + phase[i]) / 365.0))
        noise = rng.normal(1.0, noise_cv, size=n)
        raw[i, :] = np.maximum(0.0, weights[i] * daily * seasonal * noise)

    denom = np.maximum(np.sum(raw, axis=0), 1e-9)
    return raw * (np.asarray(total_it_kw, dtype=float).reshape(1, -1) / denom.reshape(1, -1))


def _ensure_base_series(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    if "hour" not in df.columns:
        df["hour"] = np.arange(n)
    hour = df["hour"].to_numpy(dtype=float)
    hod = hour % 24.0
    doy = (hour // 24.0) % 365.0
    rng = np.random.default_rng(config["project"]["random_seed"])

    if "price_grid" not in df.columns:
        price = 0.55 + 0.12 * (np.sin(2 * np.pi * (hod - 8) / 24.0) + 1.0) / 2.0
        df["price_grid"] = np.clip(price + rng.normal(0.0, 0.01, size=n), 0.45, 0.8)
    if "cf_grid" not in df.columns:
        cf = 0.56 + 0.06 * (np.sin(2 * np.pi * (hod - 12) / 24.0) + 1.0) / 2.0
        df["cf_grid"] = np.clip(cf + rng.normal(0.0, 0.005, size=n), 0.45, 0.7)
    if "heat_demand_kw" not in df.columns:
        winter = np.cos(2 * np.pi * (doy - 15) / 365.0)
        daily = np.clip(np.sin(2 * np.pi * (hod - 6) / 24.0), 0.0, None)
        heat = 3500 * np.clip(winter, 0.0, None) + 900 * daily
        df["heat_demand_kw"] = np.clip(heat + rng.normal(0.0, 50.0, size=n), 0.0, None)
    if "t_wb" not in df.columns:
        t_wb = 19.0 + 7.0 * np.sin(2 * np.pi * (doy - 170) / 365.0) + 2.0 * np.sin(2 * np.pi * (hod - 5) / 24.0)
        df["t_wb"] = np.clip(t_wb + rng.normal(0.0, 0.35, size=n), 5.0, 31.0)
    return df


def _ensure_rack_load_columns(config: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    rack_cols = get_rack_load_cols(config)
    missing = [c for c in rack_cols if c not in df.columns]
    require_explicit = bool(config.get("model2", {}).get("rack_load", {}).get("require_explicit_rack_cols", False))
    if missing and require_explicit:
        raise RuntimeError(f"time_series.csv missing required rack IT load columns: {missing[:5]} ...")
    if missing:
        loads = _synthetic_rack_loads(config, _infer_total_it_kw(config, df))
        rack_df = pd.DataFrame(loads.T, columns=rack_cols, index=df.index)
        df = pd.concat([df.drop(columns=[c for c in rack_cols if c in df.columns]), rack_df], axis=1)
    rack_sum = df[rack_cols].sum(axis=1).to_numpy(dtype=float)
    df["it_total_kw"] = rack_sum
    return df


def ensure_time_series(config: Dict[str, Any], path: Path | None = None) -> pd.DataFrame:
    out_path = path or (ROOT / "data" / "time_series.csv")
    if out_path.exists():
        df = pd.read_csv(out_path)
    else:
        n = int(config["project"]["horizon_hours"])
        df = pd.DataFrame({"hour": np.arange(n)})
        df["it_total_kw"] = _synthetic_total_it(config, n)

    df = _ensure_base_series(config, df)
    df = _ensure_rack_load_columns(config, df)
    legacy_input_cols = [c for c in df.columns if c.startswith("d_b")]
    legacy_input_cols += [c for c in ("it_hd_kw", "it_ld_kw") if c in df.columns]
    if legacy_input_cols:
        df = df.drop(columns=legacy_input_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    cfg = load_config()
    df_out = ensure_time_series(cfg)
    print(f"Generated/loaded time series with {len(df_out)} rows")
