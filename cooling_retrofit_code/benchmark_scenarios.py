from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from decision import DecisionSchema, PlanningDecision
from model import EvaluationResult, evaluate_solution


EvaluateFn = Callable[[np.ndarray, DecisionSchema, dict, pd.DataFrame, dict[str, float]], EvaluationResult]


def evaluate_benchmark_scenarios(
    schema: DecisionSchema,
    config: dict,
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
    pareto: pd.DataFrame,
    evaluate_fn: EvaluateFn = evaluate_solution,
) -> pd.DataFrame:
    """Evaluate fixed benchmark strategies and append the selected Pareto knee solution."""
    rows: list[dict[str, object]] = []

    baseline_1_vector = _no_retrofit_vector(schema)
    baseline_1_frame = _no_heat_obligation_frame(time_frame) if evaluate_fn is evaluate_solution else time_frame
    baseline_1_result = evaluate_fn(
        baseline_1_vector,
        schema,
        config,
        baseline_1_frame,
        peak_load_by_zone_kw,
    )
    rows.append(
        _evaluation_row(
            "baseline_1_no_retrofit",
            "Baseline 1: no retrofit with existing redundancy",
            baseline_1_vector,
            baseline_1_result,
            baseline_1_frame,
            config,
        )
    )

    for scenario_key, scenario_label, vector in (
        (
            "baseline_2_aggressive_retrofit",
            "Baseline 2: aggressive whole-room retrofit",
            _aggressive_retrofit_vector(schema, config, time_frame, peak_load_by_zone_kw),
        ),
    ):
        result = evaluate_fn(vector, schema, config, time_frame, peak_load_by_zone_kw)
        rows.append(_evaluation_row(scenario_key, scenario_label, vector, result, time_frame, config))

    knee = choose_pareto_knee(pareto)
    if knee is not None:
        proposed_vector = _vector_from_pareto_row(knee, schema.vector_length)
        if proposed_vector is None:
            rows.append(_pareto_row(knee))
        else:
            result = evaluate_fn(proposed_vector, schema, config, time_frame, peak_load_by_zone_kw)
            rows.append(
                _evaluation_row(
                    "proposed_refined_knee",
                    "Proposed: refined retrofit knee solution",
                    proposed_vector,
                    result,
                    time_frame,
                    config,
                    source_solution_id=knee.get("solution_id", ""),
                )
            )

    return _add_payback_metrics(pd.DataFrame(rows))


def choose_pareto_knee(pareto: pd.DataFrame) -> pd.Series | None:
    if pareto.empty or not {"tlcc", "tce"}.issubset(pareto.columns):
        return None
    finite = pareto.copy()
    finite["tlcc_num"] = pd.to_numeric(finite["tlcc"], errors="coerce")
    finite["tce_num"] = pd.to_numeric(finite["tce"], errors="coerce")
    finite = finite[np.isfinite(finite["tlcc_num"]) & np.isfinite(finite["tce_num"])]
    if finite.empty:
        return None
    if len(finite) < 3:
        return finite.loc[finite["tlcc_num"].idxmin()]

    tlcc_span = finite["tlcc_num"].max() - finite["tlcc_num"].min()
    tce_span = finite["tce_num"].max() - finite["tce_num"].min()
    if tlcc_span <= 0.0 or tce_span <= 0.0:
        return finite.loc[finite["tlcc_num"].idxmin()]

    norm_cost = (finite["tlcc_num"] - finite["tlcc_num"].min()) / tlcc_span
    norm_carbon = (finite["tce_num"] - finite["tce_num"].min()) / tce_span
    return finite.loc[(norm_cost.pow(2) + norm_carbon.pow(2)).idxmin()]


def _no_retrofit_vector(schema: DecisionSchema) -> np.ndarray:
    vector = schema.encode_default()
    lower, upper = schema.bounds()
    zone_count = len(schema.zone_ids)
    vector[:zone_count] = 0.0
    vector[zone_count:] = 0.0
    return np.clip(vector, lower, upper)


def _no_heat_obligation_frame(time_frame: pd.DataFrame) -> pd.DataFrame:
    frame = time_frame.copy()
    if "heating_demand_kw" in frame.columns:
        frame["heating_demand_kw"] = 0.0
    return frame


def _aggressive_retrofit_vector(
    schema: DecisionSchema,
    config: dict,
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> np.ndarray:
    vector = schema.encode_default()
    lower, upper = schema.bounds()
    zone_count = len(schema.zone_ids)
    benchmark_cfg = config.get("benchmarks", {})
    config_id = int(benchmark_cfg.get("aggressive_config_id", 6))
    technology = config.get("technology", {})
    liquid = technology.get("liquid_heat_fraction", {})
    cold_plate_fraction = float(liquid.get("cold_plate", technology.get("cold_plate_heat_fraction", 0.7)))
    cold_plate_fraction = min(max(cold_plate_fraction, 0.0), 1.0)
    vector[:zone_count] = np.clip(config_id, 0, 6)

    offsets = {
        name: zone_count + index * zone_count
        for index, name in enumerate(schema.zone_capacity_names)
    }
    for zone_index, zone_id in enumerate(schema.zone_ids):
        peak = max(0.0, float(peak_load_by_zone_kw.get(zone_id, 0.0)))
        if config_id in {1, 2, 3, 6}:
            _set_capacity(vector, upper, offsets["cap_ac_new"] + zone_index, peak)
        if config_id in {4, 6}:
            _set_capacity(vector, upper, offsets["cap_cdu"] + zone_index, peak * cold_plate_fraction)
        if config_id == 5:
            _set_capacity(vector, upper, offsets["cap_rdhx"] + zone_index, peak)

    system_offset = zone_count + len(schema.zone_capacity_names) * zone_count
    system_offsets = {
        name: system_offset + index
        for index, name in enumerate(schema.system_capacity_names)
    }
    total_peak = sum(max(0.0, float(value)) for value in peak_load_by_zone_kw.values())
    heat_peak = _heating_peak_kw(time_frame)
    bess_hours = float(benchmark_cfg.get("aggressive_bess_hours", 0.0))
    tes_hours = float(benchmark_cfg.get("aggressive_tes_hours", 0.0))
    tower_multiplier = float(benchmark_cfg.get("aggressive_tower_multiplier", 1.35))

    _set_capacity(vector, upper, system_offsets["cap_ashp"], heat_peak if config_id in {2, 4, 6} else 0.0)
    _set_capacity(vector, upper, system_offsets["cap_wshp"], heat_peak if config_id in {4, 5, 6} else 0.0)
    _set_capacity(vector, upper, system_offsets["cap_bess"], total_peak * max(0.0, bess_hours))
    _set_capacity(vector, upper, system_offsets["cap_tes"], heat_peak * max(0.0, tes_hours))
    _set_capacity(vector, upper, system_offsets["delta_cap_chiller"], total_peak if config_id in {1, 2, 3, 5} else 0.0)
    _set_capacity(vector, upper, system_offsets["delta_cap_tower"], total_peak * max(0.0, tower_multiplier))

    return np.clip(vector, lower, upper)


def _set_capacity(vector: np.ndarray, upper: np.ndarray, index: int, value: float) -> None:
    vector[index] = min(max(0.0, float(value)), float(upper[index]))


def _heating_peak_kw(time_frame: pd.DataFrame) -> float:
    if "heating_demand_kw" not in time_frame.columns:
        return 0.0
    values = pd.to_numeric(time_frame["heating_demand_kw"], errors="coerce")
    values = values[np.isfinite(values)]
    return max(0.0, float(values.max())) if not values.empty else 0.0


def _evaluation_row(
    scenario_key: str,
    scenario_label: str,
    vector: np.ndarray,
    result: EvaluationResult,
    time_frame: pd.DataFrame,
    config: dict,
    source_solution_id: object = "",
) -> dict[str, object]:
    decision = result.artifacts.get("decision")
    repaired = result.artifacts.get("repaired_vector")
    row = {
        "scenario_key": scenario_key,
        "scenario_label": scenario_label,
        "source_solution_id": source_solution_id,
        "feasible": bool(result.feasible),
        "tlcc": float(result.tlcc),
        "tce": float(result.tce),
        "violation": float(result.violation),
        "reasons": ";".join(result.reasons),
        "raw_vector_json": _vector_json(vector),
        "repaired_vector_json": _vector_json(repaired),
        "config_by_zone_json": _config_json(decision),
        "zone_capacity_json": _zone_capacity_json(decision),
        "system_capacity_json": _system_capacity_json(decision),
    }
    row.update(_benchmark_metric_summary(result, time_frame, config))
    return row


def _pareto_row(row: pd.Series) -> dict[str, object]:
    return {
        "scenario_key": "proposed_refined_knee",
        "scenario_label": "Proposed: refined retrofit knee solution",
        "source_solution_id": row.get("solution_id", ""),
        "feasible": bool(row.get("feasible", True)),
        "tlcc": float(row.get("tlcc", float("nan"))),
        "tce": float(row.get("tce", float("nan"))),
        "violation": float(row.get("violation", 0.0)),
        "reasons": row.get("reasons", ""),
        "raw_vector_json": row.get("raw_vector_json", ""),
        "repaired_vector_json": row.get("repaired_vector_json", ""),
        "config_by_zone_json": row.get("config_by_zone_json", ""),
        "zone_capacity_json": row.get("zone_capacity_json", ""),
        "system_capacity_json": row.get("system_capacity_json", ""),
        "annual_it_energy_kwh": float(row.get("annual_it_energy_kwh", float("nan"))),
        "annual_facility_energy_kwh": float(row.get("annual_facility_energy_kwh", float("nan"))),
        "annual_cooling_energy_kwh": float(row.get("annual_cooling_energy_kwh", float("nan"))),
        "annual_grid_energy_kwh": float(row.get("annual_grid_energy_kwh", float("nan"))),
        "annual_heat_supply_kwh": float(row.get("annual_heat_supply_kwh", float("nan"))),
        "operational_cost_yuan": float(row.get("operational_cost_yuan", float("nan"))),
        "operational_carbon_kg": float(row.get("operational_carbon_kg", float("nan"))),
        "waste_heat_recovery_rate": float(row.get("waste_heat_recovery_rate", float("nan"))),
        "pue": float(row.get("pue", float("nan"))),
    }


def _add_payback_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add baseline-relative operating saving payback diagnostics.

    TLCC/TCE are annualized in the current model. The fixed-cost and embodied
    components below are therefore estimated from the same annualized result
    layer as TLCC/TCE minus operating cost/carbon.
    """
    output = frame.copy()
    required = {"scenario_key", "tlcc", "tce", "operational_cost_yuan", "operational_carbon_kg"}
    if output.empty or not required.issubset(output.columns):
        return output

    tlcc = pd.to_numeric(output["tlcc"], errors="coerce")
    tce = pd.to_numeric(output["tce"], errors="coerce")
    op_cost = pd.to_numeric(output["operational_cost_yuan"], errors="coerce")
    op_carbon = pd.to_numeric(output["operational_carbon_kg"], errors="coerce")

    fixed_component = (tlcc - op_cost).clip(lower=0.0)
    embodied_component = (tce - op_carbon).clip(lower=0.0)
    output["fixed_cost_component_yuan_per_year"] = fixed_component
    output["embodied_carbon_component_kg_per_year"] = embodied_component

    baseline = output[output["scenario_key"].astype(str) == "baseline_1_no_retrofit"]
    if baseline.empty:
        output["fixed_cost_increment_vs_baseline_yuan_per_year"] = float("nan")
        output["embodied_carbon_increment_vs_baseline_kg_per_year"] = float("nan")
        output["operational_cost_saving_vs_baseline_yuan_per_year"] = float("nan")
        output["operational_carbon_saving_vs_baseline_kg_per_year"] = float("nan")
        output["fixed_cost_payback_years"] = float("nan")
        output["embodied_carbon_payback_years"] = float("nan")
        return output

    baseline_index = baseline.index[0]
    baseline_fixed = _finite_or_nan(fixed_component.loc[baseline_index])
    baseline_embodied = _finite_or_nan(embodied_component.loc[baseline_index])
    baseline_op_cost = _finite_or_nan(op_cost.loc[baseline_index])
    baseline_op_carbon = _finite_or_nan(op_carbon.loc[baseline_index])

    fixed_increment = (fixed_component - baseline_fixed).clip(lower=0.0)
    embodied_increment = (embodied_component - baseline_embodied).clip(lower=0.0)
    cost_saving = baseline_op_cost - op_cost
    carbon_saving = baseline_op_carbon - op_carbon

    output["fixed_cost_increment_vs_baseline_yuan_per_year"] = fixed_increment
    output["embodied_carbon_increment_vs_baseline_kg_per_year"] = embodied_increment
    output["operational_cost_saving_vs_baseline_yuan_per_year"] = cost_saving
    output["operational_carbon_saving_vs_baseline_kg_per_year"] = carbon_saving
    output["fixed_cost_payback_years"] = _payback_years(fixed_increment, cost_saving)
    output["embodied_carbon_payback_years"] = _payback_years(embodied_increment, carbon_saving)
    return output


def _finite_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _payback_years(increment: pd.Series, saving: pd.Series) -> pd.Series:
    inc = pd.to_numeric(increment, errors="coerce")
    sav = pd.to_numeric(saving, errors="coerce")
    years = pd.Series(np.nan, index=inc.index, dtype=float)

    zero_increment = np.isfinite(inc) & (inc <= 0.0)
    years.loc[zero_increment] = 0.0

    positive_increment = np.isfinite(inc) & (inc > 0.0)
    positive_saving = np.isfinite(sav) & (sav > 0.0)
    recoverable = positive_increment & positive_saving
    years.loc[recoverable] = inc.loc[recoverable] / sav.loc[recoverable]
    years.loc[positive_increment & ~positive_saving] = float("inf")
    return years


def _benchmark_metric_summary(result: EvaluationResult, time_frame: pd.DataFrame, config: dict) -> dict[str, float]:
    empty = {
        "annual_it_energy_kwh": float("nan"),
        "annual_facility_energy_kwh": float("nan"),
        "annual_cooling_energy_kwh": float("nan"),
        "annual_grid_energy_kwh": float("nan"),
        "annual_heat_supply_kwh": float("nan"),
        "operational_cost_yuan": float("nan"),
        "operational_carbon_kg": float("nan"),
        "waste_heat_recovery_rate": float("nan"),
        "pue": float("nan"),
    }
    dispatch = result.artifacts.get("dispatch") if isinstance(result.artifacts, dict) else None
    if not isinstance(dispatch, pd.DataFrame) or dispatch.empty:
        return empty

    it = _numeric_dispatch_series(dispatch, "it_load_kw")
    if it is None:
        return empty
    weights = _numeric_dispatch_series(dispatch, "day_weight")
    if weights is None:
        weights = pd.Series(np.ones(len(dispatch), dtype=float))
    cooling = _numeric_dispatch_series(dispatch, "p_cooling_kw")
    grid = _numeric_dispatch_series(dispatch, "p_grid_kw")
    heat = _numeric_dispatch_series(dispatch, "q_heat_kw")

    annual_it = float((it * weights).sum())
    if annual_it <= 0.0 or not np.isfinite(annual_it):
        return empty

    annual_cooling = float((cooling * weights).sum()) if cooling is not None else float("nan")
    annual_grid = float((grid * weights).sum()) if grid is not None else float("nan")
    annual_heat = float((heat * weights).sum()) if heat is not None else float("nan")
    if cooling is not None:
        annual_facility = float(((it + cooling) * weights).sum())
    elif grid is not None:
        annual_facility = annual_grid
    else:
        annual_facility = float("nan")
    pue = annual_facility / annual_it if np.isfinite(annual_facility) else float("nan")
    recovery_rate = annual_heat / annual_it if np.isfinite(annual_heat) else float("nan")
    operational_cost, operational_carbon = _operational_metric_summary(dispatch, time_frame, config)
    return {
        "annual_it_energy_kwh": annual_it,
        "annual_facility_energy_kwh": annual_facility,
        "annual_cooling_energy_kwh": annual_cooling,
        "annual_grid_energy_kwh": annual_grid,
        "annual_heat_supply_kwh": annual_heat,
        "operational_cost_yuan": operational_cost,
        "operational_carbon_kg": operational_carbon,
        "waste_heat_recovery_rate": float(recovery_rate),
        "pue": float(pue),
    }


def _operational_metric_summary(dispatch: pd.DataFrame, time_frame: pd.DataFrame, config: dict) -> tuple[float, float]:
    if len(dispatch) != len(time_frame):
        return float("nan"), float("nan")

    grid = _numeric_dispatch_series(dispatch, "p_grid_kw")
    heat = _numeric_dispatch_series(dispatch, "q_heat_kw")
    if grid is None:
        return float("nan"), float("nan")
    if heat is None:
        heat = pd.Series(np.zeros(len(dispatch), dtype=float))

    weights = _numeric_dispatch_series(time_frame, "day_weight")
    if weights is None:
        weights = _numeric_dispatch_series(dispatch, "day_weight")
    if weights is None:
        weights = pd.Series(np.ones(len(dispatch), dtype=float))

    price = _numeric_dispatch_series(time_frame, "price_yuan_per_kwh")
    if price is None:
        price = pd.Series(np.zeros(len(dispatch), dtype=float))
    carbon_factor = _numeric_dispatch_series(time_frame, "carbon_kg_per_kwh")
    if carbon_factor is None:
        carbon_factor = pd.Series(np.zeros(len(dispatch), dtype=float))

    econ_cfg = config.get("economics", {})
    carbon_cfg = config.get("carbon", {})
    heat_credit = float(econ_cfg.get("heat_credit_yuan_per_kwh", 0.0))
    heat_displacement = float(carbon_cfg.get("heat_displacement_kg_per_kwh", 0.0))
    hotspot_penalty = _numeric_dispatch_series(dispatch, "hotspot_penalty_yuan")
    bess_degradation = _numeric_dispatch_series(dispatch, "bess_degradation_cost_yuan")

    operating_cost = float(((grid * price - heat * heat_credit) * weights).sum())
    operating_cost += float(hotspot_penalty.sum()) if hotspot_penalty is not None else 0.0
    operating_cost += float(bess_degradation.sum()) if bess_degradation is not None else 0.0
    operating_carbon = float(((grid * carbon_factor - heat * heat_displacement) * weights).sum())
    if not bool(carbon_cfg.get("allow_negative_operational_carbon", False)):
        operating_carbon = max(0.0, operating_carbon)
    return max(0.0, operating_cost), operating_carbon


def _numeric_dispatch_series(dispatch: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in dispatch.columns:
        return None
    values = pd.to_numeric(dispatch[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.isna().all():
        return None
    return values.fillna(0.0).reset_index(drop=True)


def _vector_from_pareto_row(row: pd.Series, expected_length: int) -> np.ndarray | None:
    for field in ("raw_vector_json", "repaired_vector_json"):
        value = row.get(field, "")
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            vector = np.asarray(json.loads(value), dtype=float)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if vector.shape == (expected_length,) and np.isfinite(vector).all():
            return vector
    return None


def _vector_json(vector) -> str:
    if vector is None:
        return ""
    return json.dumps(np.asarray(vector, dtype=float).tolist(), ensure_ascii=False, separators=(",", ":"))


def _config_json(decision: PlanningDecision | None) -> str:
    return json.dumps(decision.s_z, ensure_ascii=False, sort_keys=True) if decision is not None else ""


def _zone_capacity_json(decision: PlanningDecision | None) -> str:
    if decision is None:
        return ""
    payload = {
        name: getattr(decision, name)
        for name in DecisionSchema.zone_capacity_names
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _system_capacity_json(decision: PlanningDecision | None) -> str:
    if decision is None:
        return ""
    payload = {
        name: float(getattr(decision, name))
        for name in DecisionSchema.system_capacity_names
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
