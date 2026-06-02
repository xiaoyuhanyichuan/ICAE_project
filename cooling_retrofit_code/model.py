from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from decision import DecisionSchema, PlanningDecision
from feasibility_oracle import FeasibilityOracle
from inner_dispatch import InnerDispatchMILP


@dataclass(frozen=True)
class EvaluationResult:
    feasible: bool
    tlcc: float
    tce: float
    violation: float = 0.0
    reasons: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def _crf(rate: float, years: int | float) -> float:
    years_float = float(years)
    if years_float <= 0.0:
        raise ValueError("Project life years must be positive.")
    rate_float = float(rate)
    if rate_float == 0.0:
        return 1.0 / years_float
    growth = (1.0 + rate_float) ** years_float
    return rate_float * growth / (growth - 1.0)


def _capacity_cost(decision: PlanningDecision, config: dict[str, Any]) -> tuple[float, float]:
    economics = config.get("economics", {})
    carbon = config.get("carbon", {})
    years = float(economics.get("project_life_years", 1))
    capital_recovery = _crf(float(economics.get("discount_rate", 0.0)), years)

    capex_yuan_per_kw = economics.get("capex_yuan_per_kw", {})
    fixed_om_yuan_per_kw_year = economics.get("fixed_om_yuan_per_kw_year", {})
    embodied_kg_per_kw = carbon.get("embodied_kg_per_kw", {})

    total_capex = 0.0
    annual_fixed_om = 0.0
    total_embodied = 0.0

    for capacity_name, cost_key in (
        ("cap_ac_new", "ac_new"),
        ("cap_rdhx", "rdhx"),
        ("cap_cdu", "cdu"),
    ):
        capacity_kw = sum(float(value) for value in getattr(decision, capacity_name).values())
        total_capex += capacity_kw * float(capex_yuan_per_kw.get(cost_key, 0.0))
        annual_fixed_om += capacity_kw * float(fixed_om_yuan_per_kw_year.get(cost_key, 0.0))
        total_embodied += capacity_kw * float(embodied_kg_per_kw.get(cost_key, 0.0))

    for attr_name, cost_key in (
        ("cap_ashp", "ashp"),
        ("cap_wshp", "wshp"),
        ("delta_cap_chiller", "chiller"),
        ("delta_cap_tower", "tower"),
        ("cap_tes", "tes"),
        ("cap_bess", "bess"),
    ):
        capacity_kw = float(getattr(decision, attr_name))
        total_capex += capacity_kw * float(capex_yuan_per_kw.get(cost_key, 0.0))
        annual_fixed_om += capacity_kw * float(fixed_om_yuan_per_kw_year.get(cost_key, 0.0))
        total_embodied += capacity_kw * float(embodied_kg_per_kw.get(cost_key, 0.0))

    retrofit_fixed_yuan = economics.get("retrofit_fixed_yuan", {})
    retrofit_fixed_kg = carbon.get("retrofit_fixed_kg", {})
    for zone in decision.zone_ids:
        config_key = str(decision.s_z[zone])
        total_capex += float(retrofit_fixed_yuan.get(config_key, 0.0))
        total_embodied += float(retrofit_fixed_kg.get(config_key, 0.0))

    return total_capex * capital_recovery + annual_fixed_om, total_embodied / years


def evaluate_solution(
    vector: np.ndarray,
    schema: DecisionSchema,
    config: dict[str, Any],
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> EvaluationResult:
    chiller_old_kw = _old_capacity_kw(
        config,
        ("chiller_old_kw", "old_chiller_kw", "existing_chiller_kw"),
    )
    tower_old_kw = _old_capacity_kw(
        config,
        ("tower_old_kw", "old_tower_kw", "existing_tower_kw"),
    )
    floor_weight_margin_by_zone_kg = _floor_weight_margin_by_zone(config, schema.zone_ids)
    equipment_kg_per_kw = config.get("weight", {}).get("equipment_kg_per_kw", {})
    cop_chiller = float(config.get("technology", {}).get("fitted", {}).get("cop_chiller", 3.0))
    cop_wshp = float(config.get("technology", {}).get("cop_wshp", 4.0))
    peak_heating_demand_kw = _peak_heating_demand_kw(time_frame)
    old_air_capacity_eff_by_zone_kw = _old_air_capacity_eff_by_zone(
        time_frame,
        peak_load_by_zone_kw,
        schema.zone_ids,
    )
    oracle = FeasibilityOracle(
        schema=schema,
        peak_load_by_zone_kw=peak_load_by_zone_kw,
        old_air_capacity_eff_by_zone_kw=old_air_capacity_eff_by_zone_kw,
        floor_weight_margin_by_zone_kg=floor_weight_margin_by_zone_kg,
        chiller_old_kw=chiller_old_kw,
        tower_old_kw=tower_old_kw,
        equipment_kg_per_kw=equipment_kg_per_kw,
        cop_chiller=cop_chiller,
        peak_heating_demand_kw=peak_heating_demand_kw,
        cop_wshp=cop_wshp,
        enable_infeasible_memory=bool(config.get("solver", {}).get("infeasible_memory", True)),
        config=config,
    )
    oracle_outcome = oracle.evaluate(vector)
    repaired_vector = oracle_outcome.repaired_vector
    repair_actions = oracle_outcome.repair_actions
    decision = oracle_outcome.decision
    screening = oracle_outcome.screening

    if not screening.feasible:
        return EvaluationResult(
            feasible=False,
            tlcc=float("inf"),
            tce=float("inf"),
            violation=float(screening.violation),
            reasons=list(screening.reasons),
            artifacts={
                "repair_actions": repair_actions,
                "repaired_vector": repaired_vector,
                "decision": decision,
                "screening": screening,
                "oracle": oracle_outcome,
            },
        )

    dispatch_result = InnerDispatchMILP(config).solve(
        decision=decision,
        time_frame=time_frame.reset_index(drop=True),
        peak_load_by_zone_kw=peak_load_by_zone_kw,
        chiller_old_kw=chiller_old_kw,
        tower_old_kw=tower_old_kw,
    )
    if not dispatch_result.feasible:
        oracle.remember_infeasible(repaired_vector)
        return EvaluationResult(
            feasible=False,
            tlcc=float("inf"),
            tce=float("inf"),
            violation=_dispatch_violation(dispatch_result.diagnostics),
            reasons=["inner_dispatch_infeasible"],
            artifacts={
                "repair_actions": repair_actions,
                "repaired_vector": repaired_vector,
                "decision": decision,
                "screening": screening,
                "oracle": oracle_outcome,
                "diagnostics": dispatch_result.diagnostics,
            },
        )

    annualized_capex, annualized_embodied = _capacity_cost(decision, config)
    operating_cost, operating_carbon = _operating_cost_and_carbon(
        dispatch_result.dispatch,
        time_frame.reset_index(drop=True),
        config,
    )
    weight_penalty_cost, weight_penalty_carbon = _weight_soft_penalty(screening, config)

    return EvaluationResult(
        feasible=True,
        tlcc=annualized_capex + operating_cost + weight_penalty_cost,
        tce=annualized_embodied + operating_carbon + weight_penalty_carbon,
        artifacts={
            "dispatch": dispatch_result.dispatch,
            "repair_actions": repair_actions,
            "repaired_vector": repaired_vector,
            "decision": decision,
            "screening": screening,
            "oracle": oracle_outcome,
            "diagnostics": dispatch_result.diagnostics,
            "weight_soft_penalty": {
                "soft_violation_kg": float(screening.soft_violation),
                "cost_yuan_per_year": weight_penalty_cost,
                "carbon_kg_per_year": weight_penalty_carbon,
                "reasons": list(screening.soft_reasons),
            },
        },
    )


def _old_capacity_kw(config: dict[str, Any], keys: tuple[str, ...]) -> float:
    for section_name in ("technology", "scenario"):
        section = config.get(section_name, {})
        for key in keys:
            if key in section:
                return float(section[key])
    return 0.0


def _old_air_capacity_eff_by_zone(
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
    zone_ids: list[str],
) -> dict[str, float]:
    capacities: dict[str, float] = {}
    for zone_id in zone_ids:
        col = f"old_ac_capacity_eff_{zone_id}_kw"
        if col in time_frame.columns:
            values = pd.to_numeric(time_frame[col], errors="coerce").dropna()
            if not values.empty:
                capacities[zone_id] = max(0.0, float(values.min()))
                continue
        capacities[zone_id] = max(0.0, float(peak_load_by_zone_kw.get(zone_id, 0.0)))
    return capacities


def _floor_weight_margin_by_zone(config: dict[str, Any], zone_ids: list[str]) -> dict[str, float]:
    weight = config.get("weight", {})
    for key in ("floor_weight_margin_by_zone_kg", "zone_weight_margin_by_zone_kg"):
        margins = weight.get(key)
        if isinstance(margins, dict):
            return {zone_id: float(margins.get(zone_id, 1.0e12)) for zone_id in zone_ids}

    uniform_margin = weight.get("floor_weight_margin_kg", weight.get("zone_weight_margin_kg"))
    if uniform_margin is not None:
        return {zone_id: float(uniform_margin) for zone_id in zone_ids}

    limits = weight.get("zone_weight_limit_kg_by_zone")
    current = weight.get("zone_existing_weight_kg_by_zone", {})
    if isinstance(limits, dict):
        return {
            zone_id: max(0.0, float(limits.get(zone_id, 1.0e12)) - float(current.get(zone_id, 0.0)))
            for zone_id in zone_ids
        }

    floor_limit = weight.get("floor_limit_kg_per_m2")
    if floor_limit is not None:
        area_by_zone = weight.get("zone_floor_area_m2_by_zone", {})
        uniform_area = weight.get("zone_floor_area_m2", 0.0)
        static_by_zone = weight.get("zone_existing_weight_kg_by_zone", {})
        rack_static = float(weight.get("rack_static_kg", 0.0))
        return {
            zone_id: max(
                0.0,
                float(floor_limit)
                * float(area_by_zone.get(zone_id, uniform_area) if isinstance(area_by_zone, dict) else uniform_area)
                - float(static_by_zone.get(zone_id, rack_static) if isinstance(static_by_zone, dict) else rack_static),
            )
            for zone_id in zone_ids
        }

    return {zone_id: 1.0e12 for zone_id in zone_ids}


def _peak_heating_demand_kw(time_frame: pd.DataFrame) -> float:
    if "heating_demand_kw" not in time_frame.columns:
        return 0.0
    return float(pd.to_numeric(time_frame["heating_demand_kw"], errors="raise").max())


def _dispatch_violation(diagnostics: dict[str, Any]) -> float:
    for key in ("max_violation", "violation", "constraint_violation"):
        if key in diagnostics:
            violation = float(diagnostics[key])
            if violation > 0.0:
                return violation
    return 1.0e6


def _weight_soft_penalty(screening, config: dict[str, Any]) -> tuple[float, float]:
    soft_violation_kg = max(0.0, float(getattr(screening, "soft_violation", 0.0)))
    if soft_violation_kg <= 0.0:
        return 0.0, 0.0

    soft_config = config.get("weight", {}).get("soft_constraint", {})
    if not bool(soft_config.get("enabled", True)):
        return 0.0, 0.0

    cost_per_kg_year = float(soft_config.get("cost_penalty_yuan_per_kg_year", 0.0))
    carbon_per_kg_year = float(soft_config.get("carbon_penalty_kg_per_kg_year", 0.0))
    return soft_violation_kg * cost_per_kg_year, soft_violation_kg * carbon_per_kg_year


def _operating_cost_and_carbon(
    dispatch: pd.DataFrame,
    time_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[float, float]:
    if len(dispatch) != len(time_frame):
        raise ValueError(
            "Dispatch and time_frame must have the same number of rows for objective aggregation."
        )

    economics = config.get("economics", {})
    carbon = config.get("carbon", {})

    p_grid_kw = _numeric_series(dispatch, "p_grid_kw", 0.0)
    q_heat_kw = _numeric_series(dispatch, "q_heat_kw", 0.0)
    hotspot_penalty_yuan = _numeric_series(dispatch, "hotspot_penalty_yuan", 0.0)
    bess_degradation_cost_yuan = _numeric_series(dispatch, "bess_degradation_cost_yuan", 0.0)
    price = _numeric_series(time_frame, "price_yuan_per_kwh", 0.0)
    carbon_factor = _numeric_series(time_frame, "carbon_kg_per_kwh", 0.0)
    day_weight = _day_weight(dispatch, time_frame)

    heat_credit = float(economics.get("heat_credit_yuan_per_kwh", 0.0))
    heat_displacement = float(carbon.get("heat_displacement_kg_per_kwh", 0.0))

    operating_cost = float(((p_grid_kw * price) - (q_heat_kw * heat_credit)).mul(day_weight).sum())
    operating_cost += float((hotspot_penalty_yuan + bess_degradation_cost_yuan).sum())
    operating_carbon = float(
        ((p_grid_kw * carbon_factor) - (q_heat_kw * heat_displacement)).mul(day_weight).sum()
    )
    if not bool(carbon.get("allow_negative_operational_carbon", False)):
        operating_carbon = max(0.0, operating_carbon)

    return operating_cost, operating_carbon


def _numeric_series(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="raise").astype(float).reset_index(drop=True)


def _day_weight(dispatch: pd.DataFrame, time_frame: pd.DataFrame) -> pd.Series:
    if "day_weight" in time_frame.columns:
        source = time_frame
    elif "day_weight" in dispatch.columns:
        source = dispatch
    else:
        return pd.Series([1.0] * len(time_frame), dtype=float)
    return pd.to_numeric(source["day_weight"], errors="raise").astype(float).reset_index(drop=True)
