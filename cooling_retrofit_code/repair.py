from __future__ import annotations

import numpy as np

from decision import DecisionSchema, PlanningDecision
from feasibility_oracle import ScreeningResult


_AC_CONFIGS = {1, 2, 3, 6}
_EXISTING_AC_CONFIGS = {0, 4}
_RDHX_CONFIGS = {5}
_CDU_CONFIGS = {4, 6}

_ALLOWED_CONFIGS_BY_CAPACITY = {
    "cap_ac_new": _AC_CONFIGS,
    "cap_rdhx": _RDHX_CONFIGS,
    "cap_cdu": _CDU_CONFIGS,
}
_DEFAULT_EQUIPMENT_KG_PER_KW = {
    "ac_new": 5.0,
    "rdhx": 6.0,
    "cdu": 5.0,
}
_DEFAULT_COLD_PLATE_FRACTION = 0.7
_DEFAULT_RDHX_FRACTION = 1.0


def repair_vector(
    vector: np.ndarray,
    schema: DecisionSchema,
    peak_load_by_zone_kw: dict[str, float],
    floor_weight_margin_by_zone_kg: dict[str, float],
    old_air_capacity_eff_by_zone_kw: dict[str, float] | None = None,
    equipment_kg_per_kw: dict[str, float] | None = None,
    chiller_old_kw: float = 0.0,
    tower_old_kw: float = 0.0,
    cop_chiller: float = 3.0,
    peak_heating_demand_kw: float = 0.0,
    cop_wshp: float = 4.0,
    cold_plate_fraction: float = _DEFAULT_COLD_PLATE_FRACTION,
    rdhx_fraction: float = _DEFAULT_RDHX_FRACTION,
) -> tuple[np.ndarray, list[str]]:
    peak_loads = _validate_peak_loads(schema.zone_ids, peak_load_by_zone_kw)
    old_air_capacity = _validate_old_air_capacity(
        schema.zone_ids,
        peak_loads,
        old_air_capacity_eff_by_zone_kw,
    )
    _validate_weight_margins(schema.zone_ids, floor_weight_margin_by_zone_kg)
    _equipment_weights(equipment_kg_per_kw)
    values = np.asarray(vector, dtype=float)
    if values.shape != (schema.vector_length,):
        raise ValueError(f"Expected vector length {schema.vector_length}, got {values.size}.")
    if not np.isfinite(values).all():
        raise ValueError("Decision vector must contain only finite values.")

    lower, upper = schema.bounds()
    repaired = np.clip(values.copy(), lower, upper)
    actions: list[str] = []

    zone_count = len(schema.zone_ids)
    repaired[:zone_count] = np.clip(np.rint(repaired[:zone_count]), 0, 6)
    capacity_offsets = {
        name: zone_count + idx * zone_count
        for idx, name in enumerate(schema.zone_capacity_names)
    }

    for zone_idx, zone_id in enumerate(schema.zone_ids):
        config = int(repaired[zone_idx])
        peak = peak_loads[zone_id]
        existing_eff = old_air_capacity[zone_id] if config in _EXISTING_AC_CONFIGS else 0.0

        for capacity_name in schema.zone_capacity_names:
            offset = capacity_offsets[capacity_name] + zone_idx
            if config not in _ALLOWED_CONFIGS_BY_CAPACITY[capacity_name] and repaired[offset] != 0.0:
                repaired[offset] = 0.0
                actions.append(f"{zone_id}:clear_{_action_capacity_name(capacity_name)}")

        if config in _AC_CONFIGS:
            _raise_capacity(
                repaired,
                capacity_offsets["cap_ac_new"] + zone_idx,
                min(max(0.0, peak - existing_eff), upper[capacity_offsets["cap_ac_new"] + zone_idx]),
                zone_id,
                "cap_ac_new",
                actions,
            )
        if config in _RDHX_CONFIGS:
            _raise_capacity(
                repaired,
                capacity_offsets["cap_rdhx"] + zone_idx,
                min(peak, upper[capacity_offsets["cap_rdhx"] + zone_idx]),
                zone_id,
                "cap_rdhx",
                actions,
            )
        if config in _CDU_CONFIGS:
            _raise_capacity(
                repaired,
                capacity_offsets["cap_cdu"] + zone_idx,
                min(peak, upper[capacity_offsets["cap_cdu"] + zone_idx]),
                zone_id,
                "cap_cdu",
                actions,
            )

    decision = schema.decode(repaired)
    _raise_tower_capacity(
        repaired,
        schema,
        decision,
        peak_loads,
        float(chiller_old_kw),
        float(tower_old_kw),
        float(cop_chiller),
        float(peak_heating_demand_kw),
        float(cop_wshp),
        upper,
        actions,
        cold_plate_fraction,
        rdhx_fraction,
    )

    return repaired, actions


def screen_decision(
    decision: PlanningDecision,
    peak_load_by_zone_kw: dict[str, float],
    floor_weight_margin_by_zone_kg: dict[str, float],
    chiller_old_kw: float,
    tower_old_kw: float = 0.0,
    old_air_capacity_eff_by_zone_kw: dict[str, float] | None = None,
    equipment_kg_per_kw: dict[str, float] | None = None,
    cop_chiller: float = 3.0,
    peak_heating_demand_kw: float = 0.0,
    cop_wshp: float = 4.0,
    cold_plate_fraction: float = _DEFAULT_COLD_PLATE_FRACTION,
    rdhx_fraction: float = _DEFAULT_RDHX_FRACTION,
) -> ScreeningResult:
    peak_loads = _validate_peak_loads(decision.zone_ids, peak_load_by_zone_kw)
    old_air_capacity = _validate_old_air_capacity(
        decision.zone_ids,
        peak_loads,
        old_air_capacity_eff_by_zone_kw,
    )
    weight_margins = _validate_weight_margins(decision.zone_ids, floor_weight_margin_by_zone_kg)
    equipment_weights = _equipment_weights(equipment_kg_per_kw)
    violation = 0.0
    reasons: list[str] = []
    soft_violation = 0.0
    soft_reasons: list[str] = []

    chiller_load_bound = 0.0
    rdhx_recoverable_bound = 0.0
    cdu_direct_rejection_bound = 0.0
    for zone_id in decision.zone_ids:
        zone_weight = _zone_equipment_weight(decision, zone_id, equipment_weights)
        if zone_weight > weight_margins[zone_id]:
            excess = zone_weight - weight_margins[zone_id]
            soft_violation += excess
            soft_reasons.append(f"{zone_id}:weight_margin_exceeded:{excess}")

        peak = peak_loads[zone_id]
        existing_air_capacity = old_air_capacity[zone_id] if decision.s_z[zone_id] in _EXISTING_AC_CONFIGS else 0.0
        supplied = (
            existing_air_capacity
            + decision.cap_ac_new[zone_id]
            + decision.cap_rdhx[zone_id]
            + decision.cap_cdu[zone_id]
        )
        if supplied < peak:
            shortage = peak - supplied
            violation += shortage
            reasons.append(f"{zone_id}:peak_shortage:{shortage}")

        zone_chiller, zone_rdhx_source, zone_cdu_free = _zone_peak_heat_flow_bounds(
            decision,
            zone_id,
            peak,
            cold_plate_fraction,
            rdhx_fraction,
        )
        chiller_load_bound += zone_chiller
        rdhx_recoverable_bound += zone_rdhx_source
        cdu_direct_rejection_bound += zone_cdu_free

    chiller_load_bound, cdu_direct_rejection_bound = _apply_wshp_source_absorption(
        chiller_load_bound,
        rdhx_recoverable_bound,
        cdu_direct_rejection_bound,
        float(decision.cap_wshp),
        float(peak_heating_demand_kw),
        float(cop_wshp),
    )
    chiller_capacity = float(chiller_old_kw) + decision.delta_cap_chiller
    if chiller_capacity < chiller_load_bound:
        shortage = chiller_load_bound - chiller_capacity
        violation += shortage
        reasons.append(f"chiller_peak_shortage:{shortage}")

    tower_heat_bound = _tower_heat_bound(
        chiller_load_bound,
        cdu_direct_rejection_bound,
        float(cop_chiller),
    )
    tower_capacity = float(tower_old_kw) + decision.delta_cap_tower
    if tower_capacity < tower_heat_bound:
        shortage = tower_heat_bound - tower_capacity
        violation += shortage
        reasons.append(f"tower_peak_shortage:{shortage}")

    return ScreeningResult(
        feasible=violation <= 0.0,
        violation=violation,
        reasons=reasons,
        soft_violation=soft_violation,
        soft_reasons=soft_reasons,
    )


def _raise_capacity(
    repaired: np.ndarray,
    offset: int,
    required_kw: float,
    zone_id: str,
    capacity_name: str,
    actions: list[str],
) -> None:
    if repaired[offset] < required_kw:
        repaired[offset] = required_kw
        actions.append(f"{zone_id}:raise_{_action_capacity_name(capacity_name)}_to_peak")


def _action_capacity_name(capacity_name: str) -> str:
    return capacity_name.removeprefix("cap_")


def _validate_peak_loads(
    zone_ids: list[str],
    peak_load_by_zone_kw: dict[str, float],
) -> dict[str, float]:
    peak_loads: dict[str, float] = {}
    for zone_id in zone_ids:
        if zone_id not in peak_load_by_zone_kw:
            raise ValueError(f"Missing peak load for zone {zone_id}.")
        peak = float(peak_load_by_zone_kw[zone_id])
        if not np.isfinite(peak):
            raise ValueError(f"Peak load for zone {zone_id} must be finite.")
        if peak < 0.0:
            raise ValueError(f"Peak load for zone {zone_id} must be non-negative.")
        peak_loads[zone_id] = peak
    return peak_loads


def _validate_old_air_capacity(
    zone_ids: list[str],
    peak_loads: dict[str, float],
    old_air_capacity_eff_by_zone_kw: dict[str, float] | None,
) -> dict[str, float]:
    if old_air_capacity_eff_by_zone_kw is None:
        return {zone_id: peak_loads[zone_id] for zone_id in zone_ids}

    capacities: dict[str, float] = {}
    for zone_id in zone_ids:
        value = float(old_air_capacity_eff_by_zone_kw.get(zone_id, peak_loads[zone_id]))
        if not np.isfinite(value):
            raise ValueError(f"Old AC effective capacity for zone {zone_id} must be finite.")
        capacities[zone_id] = max(0.0, value)
    return capacities


def _validate_weight_margins(
    zone_ids: list[str],
    floor_weight_margin_by_zone_kg: dict[str, float],
) -> dict[str, float]:
    margins: dict[str, float] = {}
    for zone_id in zone_ids:
        margin = float(floor_weight_margin_by_zone_kg.get(zone_id, 1.0e12))
        if not np.isfinite(margin):
            raise ValueError(f"Floor weight margin for zone {zone_id} must be finite.")
        margins[zone_id] = max(0.0, margin)
    return margins


def _equipment_weights(equipment_kg_per_kw: dict[str, float] | None) -> dict[str, float]:
    weights = dict(_DEFAULT_EQUIPMENT_KG_PER_KW)
    if equipment_kg_per_kw:
        for name, value in equipment_kg_per_kw.items():
            weights[name] = max(0.0, float(value))
    return weights


def _capacity_weight_key(capacity_name: str) -> str:
    return capacity_name.removeprefix("cap_")


def _zone_equipment_weight(
    decision: PlanningDecision,
    zone_id: str,
    equipment_weights: dict[str, float],
) -> float:
    weight = 0.0
    for capacity_name in DecisionSchema.zone_capacity_names:
        key = _capacity_weight_key(capacity_name)
        weight += float(getattr(decision, capacity_name)[zone_id]) * float(equipment_weights.get(key, 0.0))
    return weight


def _limit_zone_weight(
    repaired: np.ndarray,
    schema: DecisionSchema,
    zone_idx: int,
    zone_id: str,
    capacity_offsets: dict[str, int],
    weight_margin_kg: float,
    equipment_weights: dict[str, float],
    actions: list[str],
) -> None:
    weighted_offsets: list[tuple[int, float]] = []
    total_weight = 0.0
    for capacity_name in schema.zone_capacity_names:
        offset = capacity_offsets[capacity_name] + zone_idx
        weight_per_kw = float(equipment_weights.get(_capacity_weight_key(capacity_name), 0.0))
        if weight_per_kw <= 0.0:
            continue
        weighted_offsets.append((offset, weight_per_kw))
        total_weight += repaired[offset] * weight_per_kw

    if total_weight <= weight_margin_kg or total_weight <= 0.0:
        return

    scale = weight_margin_kg / total_weight
    for offset, _ in weighted_offsets:
        repaired[offset] *= scale
    actions.append(f"{zone_id}:scale_capacity_to_weight_margin")


def _raise_tower_capacity(
    repaired: np.ndarray,
    schema: DecisionSchema,
    decision: PlanningDecision,
    peak_loads: dict[str, float],
    chiller_old_kw: float,
    tower_old_kw: float,
    cop_chiller: float,
    peak_heating_demand_kw: float,
    cop_wshp: float,
    upper: np.ndarray,
    actions: list[str],
    cold_plate_fraction: float = _DEFAULT_COLD_PLATE_FRACTION,
    rdhx_fraction: float = _DEFAULT_RDHX_FRACTION,
) -> None:
    chiller_load_bound = 0.0
    rdhx_recoverable_bound = 0.0
    cdu_direct_rejection_bound = 0.0
    for zone_id in decision.zone_ids:
        peak = peak_loads[zone_id]
        zone_chiller, zone_rdhx_source, zone_cdu_free = _zone_peak_heat_flow_bounds(
            decision,
            zone_id,
            peak,
            cold_plate_fraction,
            rdhx_fraction,
        )
        chiller_load_bound += zone_chiller
        rdhx_recoverable_bound += zone_rdhx_source
        cdu_direct_rejection_bound += zone_cdu_free

    chiller_load_bound, cdu_direct_rejection_bound = _apply_wshp_source_absorption(
        chiller_load_bound,
        rdhx_recoverable_bound,
        cdu_direct_rejection_bound,
        float(decision.cap_wshp),
        peak_heating_demand_kw,
        cop_wshp,
    )
    system_start = len(schema.zone_ids) + len(schema.zone_capacity_names) * len(schema.zone_ids)
    try:
        chiller_idx = schema.system_capacity_names.index("delta_cap_chiller")
    except ValueError:
        chiller_idx = -1
    if chiller_idx >= 0:
        chiller_offset = system_start + chiller_idx
        required_delta_chiller = max(0.0, chiller_load_bound - chiller_old_kw)
        target = min(required_delta_chiller, upper[chiller_offset])
        if repaired[chiller_offset] < target:
            repaired[chiller_offset] = target
            actions.append("raise_chiller_to_peak_evaporator_load")

    required_tower = _tower_heat_bound(
        chiller_load_bound,
        cdu_direct_rejection_bound,
        cop_chiller,
    )
    required_delta = max(0.0, required_tower - tower_old_kw)
    if required_delta <= 0.0:
        return

    try:
        tower_idx = schema.system_capacity_names.index("delta_cap_tower")
    except ValueError:
        return
    offset = system_start + tower_idx
    target = min(required_delta, upper[offset])
    if repaired[offset] < target:
        repaired[offset] = target
        actions.append("raise_tower_to_peak_heat_rejection")


def _apply_wshp_source_absorption(
    chiller_load_bound_kw: float,
    rdhx_recoverable_bound_kw: float,
    cdu_direct_rejection_bound_kw: float,
    cap_wshp_kw: float = 0.0,
    peak_heating_demand_kw: float = 0.0,
    cop_wshp: float = 4.0,
) -> tuple[float, float]:
    heat_delivered_by_wshp = min(max(0.0, float(cap_wshp_kw)), max(0.0, float(peak_heating_demand_kw)))
    source_heat_capacity = heat_delivered_by_wshp * (1.0 - 1.0 / max(float(cop_wshp), 1.0e-9))

    rdhx_absorbed_by_wshp = min(max(0.0, float(rdhx_recoverable_bound_kw)), source_heat_capacity)
    remaining_source_heat_capacity = max(0.0, source_heat_capacity - rdhx_absorbed_by_wshp)
    cdu_absorbed_by_wshp = min(max(0.0, float(cdu_direct_rejection_bound_kw)), remaining_source_heat_capacity)

    return (
        max(0.0, float(chiller_load_bound_kw) - rdhx_absorbed_by_wshp),
        max(0.0, float(cdu_direct_rejection_bound_kw) - cdu_absorbed_by_wshp),
    )


def _zone_peak_heat_flow_bounds(
    decision: PlanningDecision,
    zone_id: str,
    peak_kw: float,
    cold_plate_fraction: float = _DEFAULT_COLD_PLATE_FRACTION,
    rdhx_fraction: float = _DEFAULT_RDHX_FRACTION,
) -> tuple[float, float, float]:
    peak = max(0.0, float(peak_kw))
    cp_fraction = min(max(float(cold_plate_fraction), 0.0), 1.0)
    rdhx_fraction = min(max(float(rdhx_fraction), 0.0), 1.0)
    cp_bound = (
        min(float(decision.cap_cdu[zone_id]), peak * cp_fraction)
        if decision.p_z.get(zone_id, 0)
        else 0.0
    )
    rdhx_bound = (
        min(float(decision.cap_rdhx[zone_id]), peak * rdhx_fraction)
        if decision.r_z.get(zone_id, 0)
        else 0.0
    )
    residual_air = max(0.0, peak - cp_bound - rdhx_bound)
    chiller_bound = residual_air + rdhx_bound
    return chiller_bound, rdhx_bound, cp_bound


def _tower_heat_bound(
    chiller_load_bound_kw: float,
    cdu_direct_rejection_bound_kw: float,
    cop_chiller: float,
) -> float:
    cop = max(float(cop_chiller), 1.0e-9)
    chiller_condenser_heat = chiller_load_bound_kw * (1.0 + 1.0 / cop)
    return chiller_condenser_heat + max(0.0, cdu_direct_rejection_bound_kw)
