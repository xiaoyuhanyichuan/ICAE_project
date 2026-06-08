from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from decision import DecisionSchema, PlanningDecision
from feasibility_oracle import FeasibilityOracle
from inner_dispatch import InnerDispatchMILP
from persistent_inner_dispatch import PersistentInnerDispatchModel
from warm_start import WarmStartManager


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


def _zone_rack_count(config: dict[str, Any], zone: str) -> float:
    scenario = config.get("scenario", {})
    for key in ("zone_rack_count", "zone_rack_counts"):
        counts = scenario.get(key, config.get(key, {}))
        if isinstance(counts, dict) and str(zone) in counts:
            return max(0.0, float(counts[str(zone)]))
    return 1.0


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

    cold_plate_capacity_kw = sum(
        float(decision.cap_cdu[zone])
        for zone in decision.zone_ids
        if int(decision.p_z.get(zone, 0)) == 1
    )
    total_capex += cold_plate_capacity_kw * float(capex_yuan_per_kw.get("cold_plate", 0.0))
    annual_fixed_om += cold_plate_capacity_kw * float(fixed_om_yuan_per_kw_year.get("cold_plate", 0.0))
    total_embodied += cold_plate_capacity_kw * float(embodied_kg_per_kw.get("cold_plate", 0.0))

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
        rack_count = _zone_rack_count(config, zone)
        total_capex += rack_count * float(retrofit_fixed_yuan.get(config_key, 0.0))
        total_embodied += rack_count * float(retrofit_fixed_kg.get(config_key, 0.0))

    return total_capex * capital_recovery + annual_fixed_om, total_embodied / years


class EvaluationContext:
    def __init__(
        self,
        schema: DecisionSchema,
        config: dict[str, Any],
        time_frame: pd.DataFrame,
        peak_load_by_zone_kw: dict[str, float],
    ) -> None:
        self.schema = schema
        self.config = config
        self.time_frame = time_frame.reset_index(drop=True)
        self.peak_load_by_zone_kw = dict(peak_load_by_zone_kw)
        self.cop_chiller = float(config.get("technology", {}).get("fitted", {}).get("cop_chiller", 3.0))
        self.chiller_old_kw, self.tower_old_kw = _old_cooling_system_capacities(
            config,
            self.peak_load_by_zone_kw,
            self.cop_chiller,
        )
        self.floor_weight_margin_by_zone_kg = _floor_weight_margin_by_zone(config, schema.zone_ids)
        self.equipment_kg_per_kw = config.get("weight", {}).get("equipment_kg_per_kw", {})
        self.cop_wshp = float(config.get("technology", {}).get("cop_wshp", 4.0))
        self.peak_heating_demand_kw = _peak_heating_demand_kw(self.time_frame)
        self.old_air_capacity_eff_by_zone_kw = _old_air_capacity_eff_by_zone(
            self.time_frame,
            self.peak_load_by_zone_kw,
            schema.zone_ids,
        )
        self.oracle = FeasibilityOracle(
            schema=schema,
            peak_load_by_zone_kw=self.peak_load_by_zone_kw,
            old_air_capacity_eff_by_zone_kw=self.old_air_capacity_eff_by_zone_kw,
            floor_weight_margin_by_zone_kg=self.floor_weight_margin_by_zone_kg,
            chiller_old_kw=self.chiller_old_kw,
            tower_old_kw=self.tower_old_kw,
            equipment_kg_per_kw=self.equipment_kg_per_kw,
            cop_chiller=self.cop_chiller,
            peak_heating_demand_kw=self.peak_heating_demand_kw,
            cop_wshp=self.cop_wshp,
            enable_infeasible_memory=bool(config.get("solver", {}).get("infeasible_memory", True)),
            config=config,
        )
        self.warm_start_manager = WarmStartManager.from_config(config, _time_frame_fingerprint(self.time_frame))
        inner = config.get("solver", {}).get("inner", {})
        self.build_mode = str(inner.get("build_mode", "fresh")) if isinstance(inner, dict) else "fresh"
        persistent = inner.get("persistent_template", {}) if isinstance(inner, dict) else {}
        persistent_enabled = bool(persistent.get("enabled", False)) if isinstance(persistent, dict) else False
        if self.build_mode == "template_reuse" or persistent_enabled:
            self.dispatch_solver = PersistentInnerDispatchModel(config)
            self.build_mode = "template_reuse"
        else:
            self.dispatch_solver = InnerDispatchMILP(config)

    def evaluate(self, vector: np.ndarray) -> EvaluationResult:
        return _evaluate_solution_with_context(vector, self)


def evaluate_solution(
    vector: np.ndarray,
    schema: DecisionSchema,
    config: dict[str, Any],
    time_frame: pd.DataFrame,
    peak_load_by_zone_kw: dict[str, float],
) -> EvaluationResult:
    return EvaluationContext(schema, config, time_frame, peak_load_by_zone_kw).evaluate(vector)


def _evaluate_solution_with_context(vector: np.ndarray, context: EvaluationContext) -> EvaluationResult:
    schema = context.schema
    config = context.config
    time_frame = context.time_frame
    peak_load_by_zone_kw = context.peak_load_by_zone_kw
    chiller_old_kw = context.chiller_old_kw
    tower_old_kw = context.tower_old_kw
    oracle = context.oracle
    oracle_outcome = oracle.evaluate(vector)
    repaired_vector = oracle_outcome.repaired_vector
    repair_actions = oracle_outcome.repair_actions
    decision = oracle_outcome.decision
    screening = oracle_outcome.screening
    screening_rejected = not bool(screening.feasible)
    bypass_hard_screening = _bypass_hard_screening_enabled(config)

    if screening_rejected and not bypass_hard_screening:
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

    warm_hit = context.warm_start_manager.lookup(decision)
    dispatch_result = context.dispatch_solver.solve(
        decision=decision,
        time_frame=time_frame.reset_index(drop=True),
        peak_load_by_zone_kw=peak_load_by_zone_kw,
        chiller_old_kw=chiller_old_kw,
        tower_old_kw=tower_old_kw,
        warm_start=warm_hit.payload if warm_hit else None,
        build_mode=context.build_mode,
    )
    dispatch_result.diagnostics["warm_start_hit"] = warm_hit is not None
    dispatch_result.diagnostics["warm_start_key_match"] = warm_hit.match_type if warm_hit else ""
    dispatch_result.diagnostics["warm_start_cache_entries"] = context.warm_start_manager.entry_count
    dispatch_result.diagnostics["warm_start_exact_hits"] = context.warm_start_manager.stats["exact_hits"]
    dispatch_result.diagnostics["warm_start_neighbor_hits"] = context.warm_start_manager.stats["neighbor_hits"]
    dispatch_result.diagnostics["warm_start_misses"] = context.warm_start_manager.stats["misses"]
    dispatch_result.diagnostics["screening_rejected"] = bool(screening_rejected)
    dispatch_result.diagnostics["screening_bypassed"] = bool(bypass_hard_screening and screening_rejected)
    dispatch_result.diagnostics["screening_reasons"] = ";".join(screening.reasons)
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

    context.warm_start_manager.remember(
        decision,
        getattr(dispatch_result, "warm_start_payload", {}),
    )
    dispatch_result.diagnostics["warm_start_cache_entries"] = context.warm_start_manager.entry_count
    dispatch_result.diagnostics["warm_start_exact_hits"] = context.warm_start_manager.stats["exact_hits"]
    dispatch_result.diagnostics["warm_start_neighbor_hits"] = context.warm_start_manager.stats["neighbor_hits"]
    dispatch_result.diagnostics["warm_start_misses"] = context.warm_start_manager.stats["misses"]
    annualized_capex, annualized_embodied = _capacity_cost(decision, config)
    operating_cost, operating_carbon = _operating_cost_and_carbon(
        dispatch_result.dispatch,
        time_frame.reset_index(drop=True),
        config,
    )
    return EvaluationResult(
        feasible=True,
        tlcc=annualized_capex + operating_cost,
        tce=annualized_embodied + operating_carbon,
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
                "cost_yuan_per_year": 0.0,
                "carbon_kg_per_year": 0.0,
                "reasons": list(screening.soft_reasons),
            },
            "weight_hard_constraint": {
                "violation_kg": _weight_hard_violation(screening),
                "reasons": [
                    reason
                    for reason in screening.reasons
                    if "weight_margin_exceeded" in str(reason)
                ],
            },
        },
    )


def _weight_hard_violation(screening: Any) -> float:
    total = 0.0
    for reason in getattr(screening, "reasons", []):
        parts = str(reason).split(":")
        if len(parts) >= 3 and parts[-2] == "weight_margin_exceeded":
            try:
                total += max(0.0, float(parts[-1]))
            except ValueError:
                continue
    return total


def _old_capacity_kw(config: dict[str, Any], keys: tuple[str, ...]) -> float:
    for section_name in ("technology", "scenario"):
        section = config.get(section_name, {})
        for key in keys:
            if key in section:
                return float(section[key])
    return 0.0


def _old_cooling_system_capacities(
    config: dict[str, Any],
    peak_load_by_zone_kw: dict[str, float],
    cop_chiller: float,
) -> tuple[float, float]:
    chiller_old_kw = _old_capacity_kw(
        config,
        ("chiller_old_kw", "old_chiller_kw", "existing_chiller_kw"),
    )
    tower_old_kw = _old_capacity_kw(
        config,
        ("tower_old_kw", "old_tower_kw", "existing_tower_kw"),
    )
    redundancy = config.get("scenario", {}).get("existing_cooling_redundancy", {})
    if not isinstance(redundancy, dict) or not bool(redundancy.get("enabled", True)):
        return chiller_old_kw, tower_old_kw

    peak_evaporator_kw = sum(max(0.0, float(value)) for value in peak_load_by_zone_kw.values())
    system_factor = max(1.0, float(redundancy.get("factor", 1.0)))
    chiller_factor = max(1.0, float(redundancy.get("chiller_factor", system_factor)))
    tower_factor = max(1.0, float(redundancy.get("tower_factor", system_factor)))
    required_chiller_kw = peak_evaporator_kw * chiller_factor
    condenser_multiplier = 1.0 + 1.0 / max(float(cop_chiller), 1.0e-6)
    required_tower_kw = peak_evaporator_kw * condenser_multiplier * tower_factor
    return max(chiller_old_kw, required_chiller_kw), max(tower_old_kw, required_tower_kw)


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


def _time_frame_fingerprint(time_frame: pd.DataFrame) -> str:
    cols = list(map(str, time_frame.columns))
    pieces: list[str] = [str(len(time_frame)), "|".join(cols)]
    for column in ("representative_day_id", "day_weight", "timestamp_hour_utc"):
        if column in time_frame.columns:
            values = time_frame[column].astype(str).head(16).tolist()
            pieces.append(f"{column}:{','.join(values)}")
    return hashlib.sha1("||".join(pieces).encode("utf-8")).hexdigest()[:16]


def _dispatch_violation(diagnostics: dict[str, Any]) -> float:
    for key in ("max_violation", "violation", "constraint_violation"):
        if key in diagnostics:
            violation = float(diagnostics[key])
            if violation > 0.0:
                return violation
    return 1.0e6


def _bypass_hard_screening_enabled(config: dict[str, Any]) -> bool:
    screening_config = config.get("solver", {}).get("screening", {})
    if not isinstance(screening_config, dict):
        return False
    return bool(screening_config.get("bypass_hard", False))


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
