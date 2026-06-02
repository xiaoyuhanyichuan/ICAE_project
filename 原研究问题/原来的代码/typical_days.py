from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class TypicalDayData:
    scenario_names: List[str]
    scenario_day_index: np.ndarray
    weights_days: np.ndarray
    is_peak: np.ndarray
    series: Dict[str, np.ndarray]
    rack_load_cols: List[str]

    @property
    def n_scenarios(self) -> int:
        return int(len(self.scenario_names))

    @property
    def horizon(self) -> int:
        return 24


class TypicalDayBuilder:
    """Build 4 seasonal typical days + 1 peak day from hourly time series."""

    def __init__(
        self,
        n_typical: int = 4,
        include_peak_day: bool = True,
        peak_weight_days: float = 1.0,
    ):
        self.n_typical = int(n_typical)
        self.include_peak_day = bool(include_peak_day)
        self.peak_weight_days = float(peak_weight_days)

    def _season_id(self, day: int) -> int:
        # 0:winter,1:spring,2:summer,3:autumn
        if day <= 58 or day >= 334:
            return 0
        if day <= 151:
            return 1
        if day <= 243:
            return 2
        return 3

    def _pick_medoid(self, feats: np.ndarray, idx: np.ndarray) -> int:
        if len(idx) == 1:
            return int(idx[0])
        x = feats[idx, :]
        center = np.mean(x, axis=0, keepdims=True)
        d = np.sum((x - center) ** 2, axis=1)
        return int(idx[int(np.argmin(d))])

    def build(self, ts: pd.DataFrame, rack_load_cols: List[str]) -> TypicalDayData:
        required = ["hour", "price_grid", "cf_grid", "heat_demand_kw", "t_wb", "it_total_kw"]
        missing = [c for c in required if c not in ts.columns]
        if missing:
            raise ValueError(f"time_series missing required columns: {missing}")
        rack_missing = [c for c in rack_load_cols if c not in ts.columns]
        if rack_missing:
            raise ValueError(f"time_series missing rack IT load columns: {rack_missing[:5]} ...")

        n = len(ts)
        if n % 24 != 0:
            raise ValueError(f"time_series length must be multiple of 24, got {n}")
        n_days = n // 24
        if n_days < 30:
            raise ValueError(f"time_series too short for typical-day extraction: {n_days} days")

        used_cols = ["price_grid", "cf_grid", "heat_demand_kw", "t_wb", "it_total_kw"] + rack_load_cols
        arr = {c: ts[c].to_numpy(dtype=float).reshape(n_days, 24) for c in used_cols}
        rack_stack = np.stack([arr[c] for c in rack_load_cols], axis=0)
        rack_p10 = np.percentile(rack_stack, 10, axis=0)
        rack_p50 = np.percentile(rack_stack, 50, axis=0)
        rack_p90 = np.percentile(rack_stack, 90, axis=0)
        rack_std = np.std(rack_stack, axis=0)

        # Feature for day clustering/selection
        feats = np.concatenate(
            [
                arr["price_grid"],
                arr["cf_grid"],
                arr["heat_demand_kw"],
                arr["t_wb"],
                arr["it_total_kw"],
                rack_p10,
                rack_p50,
                rack_p90,
                rack_std,
            ],
            axis=1,
        )
        std = np.std(feats, axis=0, keepdims=True)
        std = np.where(std < 1e-9, 1.0, std)
        feats = (feats - np.mean(feats, axis=0, keepdims=True)) / std

        day_idx = np.arange(n_days, dtype=int)
        seasons = np.array([self._season_id(int(d)) for d in day_idx], dtype=int)

        if self.n_typical != 4:
            # Keep behavior deterministic: if user changes n_typical, slice seasons by equal bins
            bins = np.array_split(day_idx, self.n_typical)
            selected = [self._pick_medoid(feats, np.array(b, dtype=int)) for b in bins if len(b) > 0]
            names = [f"typical_{i+1}" for i in range(len(selected))]
            raw_weights = np.array([len(b) for b in bins if len(b) > 0], dtype=float)
            season_of_selected = np.zeros(len(selected), dtype=int)
        else:
            selected = []
            names = ["winter_typical", "spring_typical", "summer_typical", "autumn_typical"]
            raw_weights = []
            season_of_selected = []
            for s in range(4):
                idx_s = day_idx[seasons == s]
                if len(idx_s) == 0:
                    continue
                med = self._pick_medoid(feats, idx_s)
                selected.append(med)
                raw_weights.append(float(len(idx_s)))
                season_of_selected.append(s)
            raw_weights = np.array(raw_weights, dtype=float)
            season_of_selected = np.array(season_of_selected, dtype=int)
            names = names[: len(selected)]

        # Peak day (for hard-capacity/thermal checks)
        daily_peak_score = (
            np.max(arr["heat_demand_kw"], axis=1)
            + np.max(arr["it_total_kw"], axis=1) * 0.2
        )
        peak_day = int(np.argmax(daily_peak_score))

        scenario_names = list(names)
        scenario_days = list(selected)
        weights = raw_weights.copy()
        is_peak = [False] * len(scenario_days)

        if self.include_peak_day:
            scenario_names.append("peak_day")
            scenario_days.append(peak_day)
            is_peak.append(True)

            # Keep annual day-weight consistent
            peak_w = float(np.clip(self.peak_weight_days, 0.0, 365.0))
            if peak_w > 0:
                if self.n_typical == 4 and len(season_of_selected) == len(weights):
                    s_peak = self._season_id(peak_day)
                    hit = np.where(season_of_selected == s_peak)[0]
                    if len(hit) > 0:
                        weights[hit[0]] = max(0.0, weights[hit[0]] - peak_w)
                else:
                    # For generic bins, subtract from nearest selected day bucket
                    dsel = np.abs(np.array(selected, dtype=int) - peak_day)
                    if len(dsel) > 0:
                        j = int(np.argmin(dsel))
                        weights[j] = max(0.0, weights[j] - peak_w)
            weights = np.concatenate([weights, np.array([peak_w], dtype=float)], axis=0)

        wsum = float(np.sum(weights))
        if abs(wsum - 365.0) > 1e-6 and wsum > 1e-9:
            weights *= 365.0 / wsum

        series = {c: np.stack([arr[c][d, :] for d in scenario_days], axis=0) for c in used_cols}
        return TypicalDayData(
            scenario_names=scenario_names,
            scenario_day_index=np.array(scenario_days, dtype=int),
            weights_days=weights.astype(float),
            is_peak=np.array(is_peak, dtype=bool),
            series=series,
            rack_load_cols=list(rack_load_cols),
        )
