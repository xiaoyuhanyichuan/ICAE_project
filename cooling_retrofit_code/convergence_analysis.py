from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def analyze_pymoo_convergence(
    all_evaluations: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze generation-level convergence using pymoo indicators when available.

    The outer optimizer can remain the local NSGA-II implementation. This function
    post-processes the historical feasible population by generation and applies
    Hypervolume / IGD+ to the cumulative nondominated set up to each generation.
    """

    output = Path(output_dir)
    convergence_dir = output / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = convergence_dir / "pymoo_convergence_metrics.csv"
    summary_path = convergence_dir / "pymoo_convergence_summary.json"

    config = config or {}
    options = config.get("convergence", {})
    summary: dict[str, Any] = {
        "available": False,
        "reason": "",
        "indicator_backend": "pymoo",
        "reference_front": "history_nondominated",
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
    }

    required = {"generation", "feasible", "tlcc", "tce"}
    missing = sorted(required.difference(all_evaluations.columns))
    if missing:
        summary["reason"] = "missing_columns:" + ",".join(missing)
        pd.DataFrame().to_csv(metrics_path, index=False, encoding="utf-8-sig")
        _write_summary(summary_path, summary)
        return {"summary": summary, "metrics": pd.DataFrame(), "metrics_path": metrics_path, "summary_path": summary_path}

    feasible_mask = _bool_series(all_evaluations["feasible"])
    frame = all_evaluations.loc[feasible_mask, ["generation", "tlcc", "tce"]].copy()
    frame["generation"] = pd.to_numeric(frame["generation"], errors="coerce")
    frame["tlcc"] = pd.to_numeric(frame["tlcc"], errors="coerce")
    frame["tce"] = pd.to_numeric(frame["tce"], errors="coerce")
    finite_mask = np.isfinite(frame[["generation", "tlcc", "tce"]].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[finite_mask].copy()

    if frame.empty:
        summary["reason"] = "no_feasible_finite_solutions"
        pd.DataFrame().to_csv(metrics_path, index=False, encoding="utf-8-sig")
        _write_summary(summary_path, summary)
        return {"summary": summary, "metrics": pd.DataFrame(), "metrics_path": metrics_path, "summary_path": summary_path}

    all_objectives = frame[["tlcc", "tce"]].to_numpy(dtype=float)
    reference_front = _nondominated(all_objectives)
    ideal = np.min(all_objectives, axis=0)
    nadir = np.max(all_objectives, axis=0)
    span = np.where(nadir > ideal, nadir - ideal, 1.0)
    reference_front_norm = _normalize(reference_front, ideal, span)
    reference_point = np.asarray(options.get("reference_point", [1.1, 1.1]), dtype=float)

    indicator = _IndicatorBackend(reference_front_norm, reference_point)
    rows: list[dict[str, float | int]] = []
    previous_hv: float | None = None
    previous_igd: float | None = None

    for generation in sorted(frame["generation"].astype(int).unique()):
        history = frame.loc[frame["generation"] <= generation, ["tlcc", "tce"]].to_numpy(dtype=float)
        front = _nondominated(history)
        front_norm = _normalize(front, ideal, span)
        hv = indicator.hypervolume(front_norm)
        igd_plus = indicator.igd_plus(front_norm)
        hv_improvement = 0.0 if previous_hv is None else hv - previous_hv
        igd_improvement = 0.0 if previous_igd is None else previous_igd - igd_plus
        rows.append(
            {
                "generation": int(generation),
                "n_feasible_history": int(len(history)),
                "n_front": int(len(front)),
                "hv": float(hv),
                "igd_plus": float(igd_plus),
                "hv_improvement": float(hv_improvement),
                "hv_relative_improvement": float(hv_improvement / max(abs(previous_hv or 0.0), 1.0e-12)),
                "igd_plus_improvement": float(igd_improvement),
                "igd_plus_relative_improvement": float(igd_improvement / max(abs(previous_igd or 0.0), 1.0e-12)),
            }
        )
        previous_hv = hv
        previous_igd = igd_plus

    metrics = pd.DataFrame(rows)
    recommended_generation = _recommend_generation(metrics, options)
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary.update(
        {
            "available": True,
            "reason": "",
            "indicator_backend": indicator.backend_name,
            "final_generation": int(metrics["generation"].iloc[-1]),
            "n_generation_metrics": int(len(metrics)),
            "observed_generations": [int(value) for value in metrics["generation"].tolist()],
            "recommended_generation": recommended_generation,
            "final_hv": float(metrics["hv"].iloc[-1]),
            "final_igd_plus": float(metrics["igd_plus"].iloc[-1]),
            "final_n_feasible": int(metrics["n_feasible_history"].iloc[-1]),
            "final_n_front": int(metrics["n_front"].iloc[-1]),
            "hv_relative_tolerance": float(options.get("hv_relative_tolerance", 0.005)),
            "igd_plus_relative_tolerance": float(options.get("igd_plus_relative_tolerance", 0.005)),
            "patience_generations": int(options.get("patience_generations", 5)),
            "min_generation": int(options.get("min_generation", 5)),
        }
    )
    _write_summary(summary_path, summary)
    return {"summary": summary, "metrics": metrics, "metrics_path": metrics_path, "summary_path": summary_path}


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _normalize(values: np.ndarray, ideal: np.ndarray, span: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(values, dtype=float) - ideal) / span
    return np.clip(normalized, 0.0, None)


def _nondominated(values: np.ndarray) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return points.reshape(0, 2)
    keep = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
        if not keep[i]:
            continue
        dominated_by_other = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        dominated_by_other[i] = False
        if dominated_by_other.any():
            keep[i] = False
    front = points[keep]
    order = np.lexsort((front[:, 1], front[:, 0]))
    return front[order]


class _IndicatorBackend:
    def __init__(self, reference_front: np.ndarray, reference_point: np.ndarray) -> None:
        self.reference_front = reference_front
        self.reference_point = reference_point
        self.backend_name = "fallback_2d"
        self._hv = None
        self._igd_plus = None
        try:
            from pymoo.indicators.hv import HV
            from pymoo.indicators.igd_plus import IGDPlus

            self._hv = HV(ref_point=reference_point)
            self._igd_plus = IGDPlus(reference_front)
            self.backend_name = "pymoo"
        except Exception:
            self._hv = None
            self._igd_plus = None

    def hypervolume(self, front: np.ndarray) -> float:
        front = _nondominated(np.asarray(front, dtype=float))
        if len(front) == 0:
            return 0.0
        if self._hv is not None:
            return float(self._call_indicator(self._hv, front))
        return _hypervolume_2d(front, self.reference_point)

    def igd_plus(self, front: np.ndarray) -> float:
        front = np.asarray(front, dtype=float)
        if len(front) == 0:
            return float("inf")
        if self._igd_plus is not None:
            return float(self._call_indicator(self._igd_plus, front))
        return _igd_plus(front, self.reference_front)

    @staticmethod
    def _call_indicator(indicator: Any, values: np.ndarray) -> float:
        if callable(indicator):
            return float(indicator(values))
        return float(indicator.do(values))


def _hypervolume_2d(front: np.ndarray, reference_point: np.ndarray) -> float:
    ref_x, ref_y = float(reference_point[0]), float(reference_point[1])
    clipped = np.minimum(np.asarray(front, dtype=float), reference_point)
    clipped = clipped[np.all(clipped < reference_point, axis=1)]
    if len(clipped) == 0:
        return 0.0
    front_nd = _nondominated(clipped)
    order = np.argsort(front_nd[:, 0])[::-1]
    hv = 0.0
    previous_x = ref_x
    for x, y in front_nd[order]:
        width = max(0.0, previous_x - float(x))
        height = max(0.0, ref_y - float(y))
        hv += width * height
        previous_x = min(previous_x, float(x))
    return float(hv)


def _igd_plus(front: np.ndarray, reference_front: np.ndarray) -> float:
    front = np.asarray(front, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    distances = []
    for ref in reference_front:
        diff = np.maximum(front - ref, 0.0)
        distances.append(float(np.sqrt(np.sum(diff * diff, axis=1)).min()))
    return float(np.mean(distances)) if distances else float("inf")


def _recommend_generation(metrics: pd.DataFrame, options: dict[str, Any]) -> int | None:
    if metrics.empty:
        return None
    patience = max(1, int(options.get("patience_generations", 5)))
    min_generation = int(options.get("min_generation", 5))
    hv_tol = float(options.get("hv_relative_tolerance", 0.005))
    igd_tol = float(options.get("igd_plus_relative_tolerance", 0.005))

    rows = metrics.reset_index(drop=True)
    if len(rows) <= patience:
        return None
    for idx in range(patience, len(rows)):
        generation = int(rows.loc[idx, "generation"])
        if generation < min_generation:
            continue
        window = rows.iloc[idx - patience + 1 : idx + 1]
        hv_ok = (window["hv_relative_improvement"].clip(lower=0.0) <= hv_tol).all()
        igd_ok = (window["igd_plus_relative_improvement"].clip(lower=0.0) <= igd_tol).all()
        if bool(hv_ok and igd_ok):
            return generation
    return None
