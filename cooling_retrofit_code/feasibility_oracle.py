from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from decision import DecisionSchema, PlanningDecision


@dataclass(frozen=True)
class ScreeningResult:
    feasible: bool
    violation: float = 0.0
    reasons: list[str] = field(default_factory=list)
    soft_violation: float = 0.0
    soft_reasons: list[str] = field(default_factory=list)
    skipped_lp_relaxation: bool = True


@dataclass(frozen=True)
class OracleOutcome:
    feasible: bool
    repaired_vector: np.ndarray
    decision: PlanningDecision
    repair_actions: list[str]
    screening: ScreeningResult
    cached_infeasible: bool = False


class FeasibilityOracle:
    def __init__(
        self,
        schema: DecisionSchema,
        peak_load_by_zone_kw: dict[str, float],
        floor_weight_margin_by_zone_kg: dict[str, float],
        chiller_old_kw: float,
        tower_old_kw: float = 0.0,
        old_air_capacity_eff_by_zone_kw: dict[str, float] | None = None,
        equipment_kg_per_kw: dict[str, float] | None = None,
        cop_chiller: float = 3.0,
        peak_heating_demand_kw: float = 0.0,
        cop_wshp: float = 4.0,
        enable_infeasible_memory: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.schema = schema
        self.peak_load_by_zone_kw = peak_load_by_zone_kw
        self.old_air_capacity_eff_by_zone_kw = old_air_capacity_eff_by_zone_kw
        self.floor_weight_margin_by_zone_kg = floor_weight_margin_by_zone_kg
        self.chiller_old_kw = float(chiller_old_kw)
        self.tower_old_kw = float(tower_old_kw)
        self.equipment_kg_per_kw = equipment_kg_per_kw
        self.cop_chiller = float(cop_chiller)
        self.peak_heating_demand_kw = float(peak_heating_demand_kw)
        self.cop_wshp = float(cop_wshp)
        self.enable_infeasible_memory = bool(enable_infeasible_memory)
        self.config = config or {}
        liquid = self.config.get("technology", {}).get("liquid_heat_fraction", {})
        tech = self.config.get("technology", {})
        self.cold_plate_fraction = float(liquid.get("cold_plate", tech.get("cold_plate_heat_fraction", 0.7)))
        self.rdhx_fraction = float(liquid.get("rdhx", 1.0))
        weight_config_raw = self.config.get("weight", {})
        weight_config = weight_config_raw if isinstance(weight_config_raw, dict) else {}
        hard_weight_raw = weight_config.get("hard_constraint", {})
        hard_weight = hard_weight_raw if isinstance(hard_weight_raw, dict) else {}
        self.weight_abs_tolerance_kg = float(
            hard_weight.get("abs_tolerance_kg", weight_config.get("abs_tolerance_kg", 1.0e-6))
        )
        self.weight_relative_tolerance = float(
            hard_weight.get("relative_tolerance", weight_config.get("relative_tolerance", 1.0e-9))
        )
        screening_config = self.config.get("solver", {}).get("screening", {})
        disabled = screening_config.get("disabled_hard_checks", []) if isinstance(screening_config, dict) else []
        self.disabled_hard_checks = {str(item) for item in disabled}
        self._infeasible_memory: set[tuple[float, ...]] = set()

    def evaluate(self, vector: np.ndarray) -> OracleOutcome:
        from repair import repair_vector, screen_decision

        repaired_vector, repair_actions = repair_vector(
            vector,
            self.schema,
            self.peak_load_by_zone_kw,
            self.floor_weight_margin_by_zone_kg,
            old_air_capacity_eff_by_zone_kw=self.old_air_capacity_eff_by_zone_kw,
            equipment_kg_per_kw=self.equipment_kg_per_kw,
            chiller_old_kw=self.chiller_old_kw,
            tower_old_kw=self.tower_old_kw,
            cop_chiller=self.cop_chiller,
            peak_heating_demand_kw=self.peak_heating_demand_kw,
            cop_wshp=self.cop_wshp,
            cold_plate_fraction=self.cold_plate_fraction,
            rdhx_fraction=self.rdhx_fraction,
            weight_abs_tolerance_kg=self.weight_abs_tolerance_kg,
            weight_relative_tolerance=self.weight_relative_tolerance,
        )
        decision = self.schema.decode(repaired_vector)
        signature = self._signature(repaired_vector)
        if self.enable_infeasible_memory and signature in self._infeasible_memory:
            screening = ScreeningResult(
                feasible=False,
                violation=1.0,
                reasons=["cached_infeasible"],
            )
            return OracleOutcome(
                feasible=False,
                repaired_vector=repaired_vector,
                decision=decision,
                repair_actions=repair_actions,
                screening=screening,
                cached_infeasible=True,
            )

        screening = screen_decision(
            decision,
            self.peak_load_by_zone_kw,
            self.floor_weight_margin_by_zone_kg,
            chiller_old_kw=self.chiller_old_kw,
            tower_old_kw=self.tower_old_kw,
            old_air_capacity_eff_by_zone_kw=self.old_air_capacity_eff_by_zone_kw,
            equipment_kg_per_kw=self.equipment_kg_per_kw,
            cop_chiller=self.cop_chiller,
            peak_heating_demand_kw=self.peak_heating_demand_kw,
            cop_wshp=self.cop_wshp,
            cold_plate_fraction=self.cold_plate_fraction,
            rdhx_fraction=self.rdhx_fraction,
            disabled_hard_checks=self.disabled_hard_checks,
            weight_abs_tolerance_kg=self.weight_abs_tolerance_kg,
            weight_relative_tolerance=self.weight_relative_tolerance,
        )
        if self.enable_infeasible_memory and not screening.feasible:
            self._infeasible_memory.add(signature)
        return OracleOutcome(
            feasible=screening.feasible,
            repaired_vector=repaired_vector,
            decision=decision,
            repair_actions=repair_actions,
            screening=screening,
        )

    def remember_infeasible(self, repaired_vector: np.ndarray) -> None:
        if self.enable_infeasible_memory:
            self._infeasible_memory.add(self._signature(repaired_vector))

    @staticmethod
    def _signature(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(vector, dtype=float), decimals=6).tolist())


def run_lp_relaxation_screening() -> ScreeningResult:
    return ScreeningResult(
        feasible=True,
        reasons=["lp_relaxation_skipped"],
        skipped_lp_relaxation=True,
    )
