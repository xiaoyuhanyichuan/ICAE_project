from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from decision import PlanningDecision
from inner_dispatch import (
    InnerDispatchMILP,
    InnerSolveResult,
    _add_synthetic_racks,
    _apply_warm_start,
    _cyclic_groups,
    _gurobi_diagnostics,
    _gurobi_solver_config,
    _heat_adjacency,
    _heat_adjacency_edge_count,
    _heat_influence_from_adjacency,
    _heat_matrix,
    _hotspot_constraints_enabled,
    _model_sections,
    _positive_float,
    _rack_load_columns,
    _rack_zone_map,
    _solution_value,
    _sum_columns,
    _supply_temp,
    _time_series_array,
)


class PersistentInnerDispatchModel:
    """Worker-local Gurobi template for the inner dispatch MILP.

    The template keeps the same mathematical structure as InnerDispatchMILP.
    Candidate planning decisions are injected by updating bounds, RHS values,
    and a small number of storage-limit coefficients.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._fallback = InnerDispatchMILP(config)
        self._model: Any | None = None
        self._gp: Any | None = None
        self._grb: Any | None = None
        self._fingerprint: tuple[Any, ...] | None = None
        self._template_build_once_s = 0.0
        self._solve_count = 0
        self._strict = bool(
            config.get("solver", {})
            .get("inner", {})
            .get("persistent_template", {})
            .get("strict", False)
        )

    def solve(
        self,
        decision: PlanningDecision,
        time_frame: pd.DataFrame,
        peak_load_by_zone_kw: dict[str, float],
        chiller_old_kw: float = 0.0,
        tower_old_kw: float = 0.0,
        warm_start: dict[str, Any] | None = None,
        build_mode: str | None = None,
    ) -> InnerSolveResult:
        solve_started = perf_counter()
        try:
            frame = time_frame.reset_index(drop=True).copy()
            fingerprint = self._make_fingerprint(decision, frame)
            template_reused = self._model is not None and fingerprint == self._fingerprint
            build_elapsed = 0.0
            if not template_reused:
                build_started = perf_counter()
                self._build_template(decision, frame)
                build_elapsed = perf_counter() - build_started
                self._template_build_once_s = build_elapsed
                self._fingerprint = fingerprint

            update_started = perf_counter()
            dynamic = self._update_dynamic_model(
                decision,
                peak_load_by_zone_kw,
                chiller_old_kw,
                tower_old_kw,
            )
            warm_start_values_applied = _apply_warm_start(
                self._vars["warm_start"],
                warm_start,
                self._data["zones"],
                self._data["times"],
            )
            update_finished = perf_counter()
            self._model.optimize()
            optimize_finished = perf_counter()

            result = self._collect_result(
                decision=decision,
                dynamic=dynamic,
                solve_started=solve_started,
                update_started=update_started,
                update_finished=update_finished,
                optimize_finished=optimize_finished,
                build_elapsed=build_elapsed,
                template_reused=template_reused,
                warm_start=warm_start,
                warm_start_values_applied=warm_start_values_applied,
                build_mode=build_mode or "template_reuse",
            )
            self._solve_count += 1
            return result
        except Exception as exc:
            if self._strict:
                raise
            result = self._fallback.solve(
                decision=decision,
                time_frame=time_frame,
                peak_load_by_zone_kw=peak_load_by_zone_kw,
                chiller_old_kw=chiller_old_kw,
                tower_old_kw=tower_old_kw,
                warm_start=warm_start,
                build_mode=build_mode or "template_reuse",
            )
            result.diagnostics["persistent_template_backend"] = "fresh_after_template_error"
            result.diagnostics["persistent_template_error"] = str(exc)
            result.diagnostics["template_reuse_hit"] = False
            result.diagnostics["template_build_once_s"] = self._template_build_once_s
            return result

    def _make_fingerprint(self, decision: PlanningDecision, frame: pd.DataFrame) -> tuple[Any, ...]:
        rack_cols = _rack_load_columns(frame)
        if not rack_cols:
            rack_cols = [f"synthetic_rack_{idx:03d}_kw" for idx, _ in enumerate(decision.zone_ids or ["zone_1"])]
        rep = tuple(frame["representative_day_id"].astype(str).tolist()) if "representative_day_id" in frame else ()
        return (tuple(decision.zone_ids), len(frame), tuple(rack_cols), rep)

    def _build_template(self, decision: PlanningDecision, frame: pd.DataFrame) -> None:
        import gurobipy as gp
        from gurobipy import GRB

        self._gp = gp
        self._grb = GRB
        if frame.empty:
            raise ValueError("empty_time_frame")

        tech = self.config.get("technology", {})
        fitted = tech.get("fitted", {})
        scenario = self.config.get("scenario", {})
        coeffs = self._coefficients(tech, fitted, scenario)

        rack_cols = _rack_load_columns(frame)
        rack_ids = [col.removesuffix("_kw") for col in rack_cols]
        if not rack_cols:
            rack_ids, rack_cols, frame = _add_synthetic_racks(frame, decision.zone_ids)

        zones = list(decision.zone_ids)
        rack_to_zone = _rack_zone_map(self.config, rack_ids, zones)
        times = list(range(len(frame)))
        rack_count = len(rack_ids)
        rack_range = range(rack_count)
        zt = [(z, t) for z in zones for t in times]
        q_it = frame[rack_cols].astype(float).to_numpy()
        heat_matrix = _heat_matrix(self.config, len(rack_ids))
        heat_adjacency = _heat_adjacency(heat_matrix, len(rack_ids))
        racks_by_zone = {
            zone: np.asarray([idx for idx, rack_zone in enumerate(rack_to_zone) if rack_zone == zone], dtype=int)
            for zone in zones
        }

        old_air_coeff_eff = {
            zone: _time_series_array(frame, f"old_ac_terminal_coeff_{zone}_kw_per_kw", coeffs["ac_fan_coeff"])
            for zone in zones
        }

        model = gp.Model("new_research_inner_dispatch_template")
        solver = _gurobi_solver_config(self.config)
        _set_gurobi_params(model, solver)

        s_air_old = model.addVars(zt, lb=0.0, name="s_air_old")
        s_air_new = model.addVars(zt, lb=0.0, name="s_air_new")
        q_air_to_ashp = model.addVars(zt, lb=0.0, name="q_air_to_ashp")
        q_cp_to_wshp = model.addVars(zt, lb=0.0, name="q_cp_to_wshp")
        q_rdhx_to_wshp = model.addVars(zt, lb=0.0, name="q_rdhx_to_wshp")
        p_grid = model.addVars(times, lb=0.0, name="p_grid")
        q_tes_ch = model.addVars(times, lb=0.0, name="q_tes_ch")
        q_tes_dis = model.addVars(times, lb=0.0, name="q_tes_dis")
        e_tes = model.addVars(times, lb=0.0, name="e_tes")
        d_tes_ch = model.addVars(times, vtype=GRB.BINARY, name="d_tes_ch")
        d_tes_dis = model.addVars(times, vtype=GRB.BINARY, name="d_tes_dis")
        p_bess_ch = model.addVars(times, lb=0.0, name="p_bess_ch")
        p_bess_dis = model.addVars(times, lb=0.0, name="p_bess_dis")
        e_bess = model.addVars(times, lb=0.0, name="e_bess")
        d_bess_ch = model.addVars(times, vtype=GRB.BINARY, name="d_bess_ch")
        d_bess_dis = model.addVars(times, vtype=GRB.BINARY, name="d_bess_dis")
        hotspot_constraints_enabled = bool(coeffs["hotspot_constraints_enabled"])
        xi = model.addVars(rack_range, times, lb=0.0, name="xi") if hotspot_constraints_enabled else None

        variables = {
            "s_air_old": s_air_old,
            "s_air_new": s_air_new,
            "q_air_to_ashp": q_air_to_ashp,
            "q_cp_to_wshp": q_cp_to_wshp,
            "q_rdhx_to_wshp": q_rdhx_to_wshp,
            "p_grid": p_grid,
            "q_tes_ch": q_tes_ch,
            "q_tes_dis": q_tes_dis,
            "e_tes": e_tes,
            "d_tes_ch": d_tes_ch,
            "d_tes_dis": d_tes_dis,
            "p_bess_ch": p_bess_ch,
            "p_bess_dis": p_bess_dis,
            "e_bess": e_bess,
            "d_bess_ch": d_bess_ch,
            "d_bess_dis": d_bess_dis,
        }
        if xi is not None:
            variables["xi"] = xi
        warm_start_vars = {name: variables[name] for name in variables if name != "xi"}

        constrs: dict[str, dict[Any, Any]] = {
            "air_ashp_enabled": {},
            "air_reject_capacity": {},
            "old_air_capacity": {},
            "new_air_capacity": {},
            "air_service_min": {},
            "air_service_capacity": {},
            "cp_wshp_upper": {},
            "cdu_capacity": {},
            "rdhx_wshp_upper": {},
            "rdhx_capacity": {},
            "ashp_capacity": {},
            "wshp_capacity": {},
            "chiller_capacity": {},
            "tower_capacity": {},
            "heat_demand": {},
            "power_balance": {},
            "tes_charge_limit": {},
            "tes_discharge_limit": {},
            "bess_charge_limit": {},
            "bess_discharge_limit": {},
            "hotspot_limit": {},
        }

        for z, t in zt:
            s_air = s_air_old[z, t] + s_air_new[z, t]
            constrs["air_ashp_enabled"][z, t] = model.addConstr(q_air_to_ashp[z, t] <= 0.0, name=f"air_ashp_enabled[{z},{t}]")
            constrs["air_reject_capacity"][z, t] = model.addConstr(-q_air_to_ashp[z, t] <= 0.0, name=f"air_reject_capacity[{z},{t}]")
            constrs["old_air_capacity"][z, t] = model.addConstr(s_air_old[z, t] <= 0.0, name=f"old_air_capacity[{z},{t}]")
            constrs["new_air_capacity"][z, t] = model.addConstr(s_air_new[z, t] <= 0.0, name=f"new_air_capacity[{z},{t}]")
            constrs["air_service_min"][z, t] = model.addConstr(s_air >= 0.0, name=f"air_service_min[{z},{t}]")
            constrs["air_service_capacity"][z, t] = model.addConstr(s_air <= 0.0, name=f"air_service_capacity[{z},{t}]")
            constrs["cp_wshp_upper"][z, t] = model.addConstr(q_cp_to_wshp[z, t] <= 0.0, name=f"cp_wshp_upper[{z},{t}]")
            constrs["cdu_capacity"][z, t] = model.addConstr(0.0 * q_cp_to_wshp[z, t] <= 0.0, name=f"cdu_capacity[{z},{t}]")
            constrs["rdhx_wshp_upper"][z, t] = model.addConstr(q_rdhx_to_wshp[z, t] <= 0.0, name=f"rdhx_wshp_upper[{z},{t}]")
            constrs["rdhx_capacity"][z, t] = model.addConstr(0.0 * q_rdhx_to_wshp[z, t] <= 0.0, name=f"rdhx_capacity[{z},{t}]")

        for t in times:
            q_air_to_ashp_sum = gp.quicksum(q_air_to_ashp[z, t] for z in zones)
            q_cp_wshp_sum = gp.quicksum(q_cp_to_wshp[z, t] for z in zones)
            q_rdhx_wshp_sum = gp.quicksum(q_rdhx_to_wshp[z, t] for z in zones)
            q_wshp_in = q_cp_wshp_sum + q_rdhx_wshp_sum
            q_ashp_out = coeffs["ashp_source_to_output"] * q_air_to_ashp_sum
            q_wshp_out = coeffs["wshp_source_to_output"] * q_wshp_in
            q_heat = q_ashp_out + q_wshp_out + q_tes_dis[t] - q_tes_ch[t]

            constrs["ashp_capacity"][t] = model.addConstr(q_ashp_out <= 0.0, name=f"ashp_capacity[{t}]")
            constrs["wshp_capacity"][t] = model.addConstr(q_wshp_out <= 0.0, name=f"wshp_capacity[{t}]")
            constrs["chiller_capacity"][t] = model.addConstr(
                -q_air_to_ashp_sum - q_rdhx_wshp_sum <= 0.0,
                name=f"chiller_capacity[{t}]",
            )
            constrs["tower_capacity"][t] = model.addConstr(
                -coeffs["tower_air_coeff"] * (q_air_to_ashp_sum + q_rdhx_wshp_sum) - q_cp_wshp_sum <= 0.0,
                name=f"tower_capacity[{t}]",
            )
            model.addConstr(q_heat >= 0.0, name=f"heat_supply_nonnegative[{t}]")
            constrs["heat_demand"][t] = model.addConstr(q_heat <= 0.0, name=f"heat_demand[{t}]")

            power_expr = p_grid[t] + p_bess_dis[t] - p_bess_ch[t]
            for z in zones:
                power_expr += -coeffs["coef_air_power"] * q_air_to_ashp[z, t]
                power_expr += -coeffs["coef_rdhx_power"] * q_rdhx_to_wshp[z, t]
                power_expr += -coeffs["coef_cp_power"] * q_cp_to_wshp[z, t]
                power_expr += -float(old_air_coeff_eff[z][t]) * s_air_old[z, t]
                power_expr += -coeffs["ac_fan_coeff"] * s_air_new[z, t]
            constrs["power_balance"][t] = model.addConstr(power_expr == 0.0, name=f"power_balance[{t}]")

            constrs["tes_charge_limit"][t] = model.addConstr(q_tes_ch[t] - d_tes_ch[t] <= 0.0, name=f"tes_charge_limit[{t}]")
            constrs["tes_discharge_limit"][t] = model.addConstr(q_tes_dis[t] - d_tes_dis[t] <= 0.0, name=f"tes_discharge_limit[{t}]")
            model.addConstr(d_tes_ch[t] + d_tes_dis[t] <= 1.0, name=f"tes_charge_discharge_mutex[{t}]")
            constrs["bess_charge_limit"][t] = model.addConstr(p_bess_ch[t] - d_bess_ch[t] <= 0.0, name=f"bess_charge_limit[{t}]")
            constrs["bess_discharge_limit"][t] = model.addConstr(p_bess_dis[t] - d_bess_dis[t] <= 0.0, name=f"bess_discharge_limit[{t}]")
            model.addConstr(d_bess_ch[t] + d_bess_dis[t] <= 1.0, name=f"bess_charge_discharge_mutex[{t}]")

        for day_indices in _cyclic_groups(frame):
            for pos, t in enumerate(day_indices):
                prev = day_indices[pos - 1]
                model.addConstr(
                    e_tes[t]
                    == (1.0 - coeffs["tes_loss"]) * e_tes[prev]
                    + coeffs["tes_eta_ch"] * q_tes_ch[t]
                    - q_tes_dis[t] / coeffs["tes_eta_dis"],
                    name=f"tes_soc[{t}]",
                )
                model.addConstr(
                    e_bess[t]
                    == (1.0 - coeffs["bess_loss"]) * e_bess[prev]
                    + coeffs["bess_eta_ch"] * p_bess_ch[t]
                    - p_bess_dis[t] / coeffs["bess_eta_dis"],
                    name=f"bess_soc[{t}]",
                )

        if hotspot_constraints_enabled and xi is not None:
            for i in rack_range:
                zone = rack_to_zone[i]
                for t in times:
                    constrs["hotspot_limit"][i, t] = model.addConstr(
                        -coeffs["beta_air"] * s_air_old[zone, t]
                        - coeffs["beta_air"] * s_air_new[zone, t]
                        - xi[i, t]
                        <= 0.0,
                        name=f"hotspot_limit[{i},{t}]",
                    )

        day_weight = frame.get("day_weight", pd.Series(1.0, index=frame.index)).astype(float).to_numpy()
        price = frame.get("price_yuan_per_kwh", pd.Series(0.0, index=frame.index)).astype(float).to_numpy()
        objective = gp.LinExpr()
        for t in times:
            q_air_to_ashp_sum = gp.quicksum(q_air_to_ashp[z, t] for z in zones)
            q_wshp_in = gp.quicksum(q_cp_to_wshp[z, t] + q_rdhx_to_wshp[z, t] for z in zones)
            q_heat = (
                coeffs["ashp_source_to_output"] * q_air_to_ashp_sum
                + coeffs["wshp_source_to_output"] * q_wshp_in
                + q_tes_dis[t]
                - q_tes_ch[t]
            )
            objective += float(day_weight[t]) * (
                float(price[t]) * p_grid[t]
                - coeffs["heat_credit"] * q_heat
                + coeffs["bess_deg_cost"] * p_bess_dis[t]
                + (
                    coeffs["hotspot_penalty"] * gp.quicksum(xi[i, t] for i in rack_range)
                    if hotspot_constraints_enabled and xi is not None
                    else 0.0
                )
            )
        model.setObjective(objective, GRB.MINIMIZE)
        model.update()

        self._model = model
        self._data = {
            "frame": frame,
            "zones": zones,
            "times": times,
            "zt": zt,
            "rack_ids": rack_ids,
            "rack_cols": rack_cols,
            "rack_to_zone": rack_to_zone,
            "rack_range": list(rack_range),
            "racks_by_zone": racks_by_zone,
            "q_it": q_it,
            "heat_matrix": heat_matrix,
            "heat_adjacency": heat_adjacency,
            "old_air_coeff_eff": old_air_coeff_eff,
            "day_weight": day_weight,
            "price": price,
            "heat_demand": frame.get("heating_demand_kw", pd.Series(0.0, index=frame.index)).astype(float).to_numpy(),
            "it_total": frame.get("it_load_kw", frame[rack_cols].sum(axis=1)).astype(float).to_numpy(),
        }
        self._coeffs = coeffs
        self._vars = {**variables, "warm_start": warm_start_vars}
        self._constrs = constrs
        self._solver_config = solver

    def _coefficients(self, tech: dict[str, Any], fitted: dict[str, Any], scenario: dict[str, Any]) -> dict[str, float]:
        cop_chiller = _positive_float(fitted.get("cop_chiller", tech.get("cop_chiller", 3.0)), 3.0)
        cop_ashp = _positive_float(tech.get("cop_ashp", 3.2), 3.2)
        cop_wshp = _positive_float(tech.get("cop_wshp", 4.0), 4.0)
        ac_fan_coeff = max(0.0, float(fitted.get("ac_fan_kw_per_kw", tech.get("ac_fan_kw_per_kw", 0.2137))))
        rdhx_coeff = max(0.0, float(fitted.get("e_rdhx_kw_per_kw", tech.get("e_rdhx_kw_per_kw", 0.0265))))
        tower_coeff = max(0.0, float(tech.get("tower_kw_per_kw_heat", 0.015)))
        cdu_free_coeff = max(0.0, float(tech.get("cdu_free_kw_per_kw", tech.get("cdu_pump_kw_per_kw", 0.02))))
        cdu_pump_coeff = max(0.0, float(tech.get("cdu_pump_kw_per_kw", 0.02)))
        wshp_pump_coeff = max(0.0, float(tech.get("wshp_pump_kw_per_kw", 0.01)))
        liquid = tech.get("liquid_heat_fraction", {})
        cp_fraction = min(max(float(liquid.get("cold_plate", tech.get("cold_plate_heat_fraction", 0.7))), 0.0), 1.0)
        rdhx_fraction = min(max(float(liquid.get("rdhx", 1.0)), 0.0), 1.0)
        thermal = tech.get("thermal", {})
        tes = tech.get("tes", {})
        bess = tech.get("bess", {})
        rte = min(max(float(bess.get("roundtrip_efficiency", tech.get("bess_roundtrip_efficiency", 0.9))), 1.0e-6), 1.0)
        ashp_source_to_output = cop_ashp / max(cop_ashp - 1.0, 1.0e-9)
        wshp_source_to_output = cop_wshp / max(cop_wshp - 1.0, 1.0e-9)
        tower_air_coeff = 1.0 + 1.0 / cop_chiller
        coef_air_power = -1.0 / cop_chiller - tower_coeff * tower_air_coeff + 1.0 / max(cop_ashp - 1.0, 1.0e-9)
        coef_rdhx_power = -1.0 / cop_chiller - tower_coeff * tower_air_coeff + 1.0 / max(cop_wshp - 1.0, 1.0e-9) + wshp_pump_coeff
        coef_cp_power = -tower_coeff + 1.0 / max(cop_wshp - 1.0, 1.0e-9) + wshp_pump_coeff - cdu_free_coeff
        return {
            "cop_chiller": cop_chiller,
            "cop_ashp": cop_ashp,
            "cop_wshp": cop_wshp,
            "ac_fan_coeff": ac_fan_coeff,
            "rdhx_coeff": rdhx_coeff,
            "tower_coeff": tower_coeff,
            "cdu_free_coeff": cdu_free_coeff,
            "cdu_pump_coeff": cdu_pump_coeff,
            "wshp_pump_coeff": wshp_pump_coeff,
            "heat_credit": float(self.config.get("economics", {}).get("heat_credit_yuan_per_kwh", 0.0)),
            "hotspot_penalty": float(scenario.get("thermal_slack_penalty_yuan_per_c_hour", scenario.get("thermal_slack_penalty", 1.0e6))),
            "hotspot_constraints_enabled": _hotspot_constraints_enabled(self.config),
            "bess_deg_cost": float(tech.get("bess_degradation_yuan_per_kwh", 0.0)),
            "cp_fraction": cp_fraction,
            "rdhx_fraction": rdhx_fraction,
            "old_supply_temp_c": float(thermal.get("old_supply_temp_c", 24.0)),
            "new_supply_temp_c": float(thermal.get("new_supply_temp_c", 22.0)),
            "aux_supply_temp_c": float(thermal.get("aux_supply_temp_c", 23.0)),
            "rack_temp_max_c": float(thermal.get("rack_temp_max_c", 27.0)),
            "beta_air": float(thermal.get("beta_air_c_per_kw", 0.02)),
            "tes_loss": min(max(float(tes.get("loss_per_hour", tech.get("tes_loss_per_hour", 0.002))), 0.0), 1.0),
            "tes_eta_ch": min(max(float(tes.get("charge_efficiency", 0.95)), 1.0e-6), 1.0),
            "tes_eta_dis": min(max(float(tes.get("discharge_efficiency", 0.95)), 1.0e-6), 1.0),
            "tes_power_to_energy": max(0.0, float(tes.get("power_to_energy", 0.5))),
            "bess_eta_ch": min(max(float(bess.get("charge_efficiency", np.sqrt(rte))), 1.0e-6), 1.0),
            "bess_eta_dis": min(max(float(bess.get("discharge_efficiency", np.sqrt(rte))), 1.0e-6), 1.0),
            "bess_loss": min(max(float(bess.get("self_discharge_per_hour", 0.0)), 0.0), 1.0),
            "bess_soc_min": min(max(float(bess.get("soc_min", 0.1)), 0.0), 1.0),
            "bess_soc_max": min(max(float(bess.get("soc_max", 0.9)), min(max(float(bess.get("soc_min", 0.1)), 0.0), 1.0)), 1.0),
            "bess_c_rate": max(0.0, float(bess.get("c_rate_per_hour", 0.5))),
            "ashp_source_to_output": ashp_source_to_output,
            "wshp_source_to_output": wshp_source_to_output,
            "tower_air_coeff": tower_air_coeff,
            "coef_air_power": coef_air_power,
            "coef_rdhx_power": coef_rdhx_power,
            "coef_cp_power": coef_cp_power,
        }

    def _update_dynamic_model(
        self,
        decision: PlanningDecision,
        peak_load_by_zone_kw: dict[str, float],
        chiller_old_kw: float,
        tower_old_kw: float,
    ) -> dict[str, Any]:
        model = self._model
        data = self._data
        coeffs = self._coeffs
        vars_ = self._vars
        constrs = self._constrs
        frame = data["frame"]
        zones = data["zones"]
        times = data["times"]
        q_it = data["q_it"]
        rack_to_zone = data["rack_to_zone"]
        racks_by_zone = data["racks_by_zone"]
        time_count = len(times)

        q_cp = np.zeros_like(q_it)
        q_rdhx = np.zeros_like(q_it)
        q_air = np.zeros_like(q_it)
        for ridx, zone_id in enumerate(rack_to_zone):
            raw = np.maximum(q_it[:, ridx], 0.0)
            cp = coeffs["cp_fraction"] * raw if decision.p_z.get(zone_id, 0) else np.zeros_like(raw)
            rdhx = coeffs["rdhx_fraction"] * raw if decision.r_z.get(zone_id, 0) else np.zeros_like(raw)
            scale = np.divide(raw, cp + rdhx, out=np.ones_like(raw), where=(cp + rdhx) > raw)
            q_cp[:, ridx] = cp * scale
            q_rdhx[:, ridx] = rdhx * scale
            q_air[:, ridx] = np.maximum(0.0, raw - q_cp[:, ridx] - q_rdhx[:, ridx])

        q_air_zone = {zone: _sum_columns(q_air, racks_by_zone[zone], time_count) for zone in zones}
        q_cp_zone = {zone: _sum_columns(q_cp, racks_by_zone[zone], time_count) for zone in zones}
        q_rdhx_zone = {zone: _sum_columns(q_rdhx, racks_by_zone[zone], time_count) for zone in zones}
        old_air_capacity_eff = {
            zone: _time_series_array(frame, f"old_ac_capacity_eff_{zone}_kw", peak_load_by_zone_kw.get(zone, 0.0))
            for zone in zones
        }
        psi_amb = {zone: _time_series_array(frame, f"psi_amb_{zone}", 1.0) for zone in zones}
        old_air_capacity = {
            zone: (
                np.maximum(0.0, old_air_capacity_eff[zone] * psi_amb[zone])
                if decision.o_z.get(zone, 0)
                else np.zeros(time_count, dtype=float)
            )
            for zone in zones
        }
        new_air_capacity = {zone: float(decision.cap_ac_new[zone]) for zone in zones}
        air_capacity = {zone: old_air_capacity[zone] + new_air_capacity[zone] for zone in zones}
        heat_influence = _heat_influence_from_adjacency(q_air, data["heat_adjacency"])
        supply_temp_by_rack = np.asarray(
            [
                _supply_temp(
                    decision,
                    zone_id,
                    coeffs["old_supply_temp_c"],
                    coeffs["new_supply_temp_c"],
                    coeffs["aux_supply_temp_c"],
                )
                for zone_id in rack_to_zone
            ],
            dtype=float,
        )

        for z in zones:
            for t in times:
                constrs["air_ashp_enabled"][z, t].RHS = int(decision.a_z.get(z, 0)) * float(q_air_zone[z][t])
                constrs["air_reject_capacity"][z, t].RHS = float(air_capacity[z][t] - q_air_zone[z][t])
                constrs["old_air_capacity"][z, t].RHS = float(old_air_capacity[z][t])
                constrs["new_air_capacity"][z, t].RHS = max(0.0, new_air_capacity[z])
                constrs["air_service_min"][z, t].RHS = float(q_air_zone[z][t])
                constrs["air_service_capacity"][z, t].RHS = float(air_capacity[z][t])
                constrs["cp_wshp_upper"][z, t].RHS = float(q_cp_zone[z][t])
                constrs["cdu_capacity"][z, t].RHS = float(decision.cap_cdu[z] - q_cp_zone[z][t] + 1.0e-7)
                constrs["rdhx_wshp_upper"][z, t].RHS = float(q_rdhx_zone[z][t])
                constrs["rdhx_capacity"][z, t].RHS = float(decision.cap_rdhx[z] - q_rdhx_zone[z][t] + 1.0e-7)

        tes_power_ub = max(0.0, decision.cap_tes * coeffs["tes_power_to_energy"])
        bess_charge_limit_kw = max(0.0, decision.cap_bess * coeffs["bess_c_rate"])
        for t in times:
            vars_["q_tes_ch"][t].UB = tes_power_ub
            vars_["q_tes_dis"][t].UB = tes_power_ub
            vars_["e_tes"][t].LB = 0.0
            vars_["e_tes"][t].UB = max(0.0, decision.cap_tes)
            vars_["p_bess_ch"][t].UB = bess_charge_limit_kw
            vars_["p_bess_dis"][t].UB = bess_charge_limit_kw
            vars_["e_bess"][t].LB = max(0.0, decision.cap_bess * coeffs["bess_soc_min"])
            vars_["e_bess"][t].UB = max(0.0, decision.cap_bess * coeffs["bess_soc_max"])
            model.chgCoeff(constrs["tes_charge_limit"][t], vars_["d_tes_ch"][t], -tes_power_ub)
            model.chgCoeff(constrs["tes_discharge_limit"][t], vars_["d_tes_dis"][t], -tes_power_ub)
            model.chgCoeff(constrs["bess_charge_limit"][t], vars_["d_bess_ch"][t], -bess_charge_limit_kw)
            model.chgCoeff(constrs["bess_discharge_limit"][t], vars_["d_bess_dis"][t], -bess_charge_limit_kw)

            air_total = sum(float(q_air_zone[z][t]) for z in zones)
            cp_total = sum(float(q_cp_zone[z][t]) for z in zones)
            rdhx_total = sum(float(q_rdhx_zone[z][t]) for z in zones)
            constrs["ashp_capacity"][t].RHS = max(0.0, decision.cap_ashp)
            constrs["wshp_capacity"][t].RHS = max(0.0, decision.cap_wshp)
            constrs["chiller_capacity"][t].RHS = max(0.0, chiller_old_kw + decision.delta_cap_chiller) - air_total - rdhx_total
            constrs["tower_capacity"][t].RHS = (
                max(0.0, tower_old_kw + decision.delta_cap_tower)
                - coeffs["tower_air_coeff"] * (air_total + rdhx_total)
                - cp_total
            )
            constrs["heat_demand"][t].RHS = max(0.0, float(data["heat_demand"][t]))
            power_const = (
                float(data["it_total"][t])
                + (air_total + rdhx_total) / coeffs["cop_chiller"]
                + coeffs["tower_coeff"] * (coeffs["tower_air_coeff"] * (air_total + rdhx_total) + cp_total)
                + coeffs["cdu_pump_coeff"] * cp_total
                + coeffs["cdu_free_coeff"] * cp_total
                + coeffs["rdhx_coeff"] * rdhx_total
            )
            constrs["power_balance"][t].RHS = power_const

        if coeffs["hotspot_constraints_enabled"]:
            for i in data["rack_range"]:
                for t in times:
                    constrs["hotspot_limit"][i, t].RHS = (
                        coeffs["rack_temp_max_c"] - float(supply_temp_by_rack[i]) - float(heat_influence[t, i])
                    )
        model.update()
        return {
            "q_air_zone": q_air_zone,
            "q_cp_zone": q_cp_zone,
            "q_rdhx_zone": q_rdhx_zone,
            "q_air": q_air,
            "q_cp": q_cp,
            "q_rdhx": q_rdhx,
            "old_air_capacity": old_air_capacity,
            "air_capacity": air_capacity,
            "heat_influence": heat_influence,
            "supply_temp_by_rack": supply_temp_by_rack,
        }

    def _collect_result(
        self,
        decision: PlanningDecision,
        dynamic: dict[str, Any],
        solve_started: float,
        update_started: float,
        update_finished: float,
        optimize_finished: float,
        build_elapsed: float,
        template_reused: bool,
        warm_start: dict[str, Any] | None,
        warm_start_values_applied: int,
        build_mode: str,
    ) -> InnerSolveResult:
        model = self._model
        vars_ = self._vars
        data = self._data
        coeffs = self._coeffs
        times = data["times"]
        zones = data["zones"]
        rack_range = data["rack_range"]
        frame = data["frame"]

        base_diag = {
            **_gurobi_diagnostics(self._solver_config),
            "model_sections": _model_sections(),
            "build_mode": build_mode,
            "warm_start_attempted": bool(warm_start),
            "warm_start_values_applied": warm_start_values_applied,
            "inner_build_total_s": update_finished - solve_started,
            "inner_update_s": update_finished - update_started,
            "inner_optimize_s": optimize_finished - update_finished,
            "template_build_once_s": build_elapsed,
            "template_reuse_hit": template_reused,
            "persistent_template_backend": "gurobi_template",
            "persistent_template_solve_count": self._solve_count + 1,
            "hotspot_constraints_enabled": bool(coeffs["hotspot_constraints_enabled"]),
        }
        if model.SolCount <= 0:
            return InnerSolveResult(
                feasible=False,
                objective_value=float("inf"),
                dispatch=pd.DataFrame(),
                diagnostics={
                    "gurobi_status": int(model.Status),
                    "reason": "no_solution",
                    **base_diag,
                },
            )

        extract_started = perf_counter()
        rows: list[dict[str, Any]] = []
        max_power_residual = 0.0
        max_heat_residual = 0.0
        max_heat_demand_upper_violation = 0.0
        for t in times:
            q_air_total = sum(float(dynamic["q_air_zone"][z][t]) for z in zones)
            q_cp_total = sum(float(dynamic["q_cp_zone"][z][t]) for z in zones)
            q_rdhx_total = sum(float(dynamic["q_rdhx_zone"][z][t]) for z in zones)
            s_air_old_total = sum(vars_["s_air_old"][z, t].X for z in zones)
            s_air_new_total = sum(vars_["s_air_new"][z, t].X for z in zones)
            q_air_to_ashp = sum(vars_["q_air_to_ashp"][z, t].X for z in zones)
            q_cp_to_wshp = sum(vars_["q_cp_to_wshp"][z, t].X for z in zones)
            q_rdhx_to_wshp = sum(vars_["q_rdhx_to_wshp"][z, t].X for z in zones)
            q_air_reject = q_air_total - q_air_to_ashp
            q_cp_free = q_cp_total - q_cp_to_wshp
            q_rdhx_chiller = q_rdhx_total - q_rdhx_to_wshp
            q_ashp_out = coeffs["ashp_source_to_output"] * q_air_to_ashp
            p_ashp = q_ashp_out / coeffs["cop_ashp"]
            q_wshp_in = q_cp_to_wshp + q_rdhx_to_wshp
            q_wshp_out = coeffs["wshp_source_to_output"] * q_wshp_in
            p_wshp = q_wshp_out / coeffs["cop_wshp"]
            q_chiller_evap = q_air_reject + q_rdhx_chiller
            p_chiller = q_chiller_evap / coeffs["cop_chiller"]
            q_tower = q_chiller_evap + p_chiller + q_cp_free
            p_tower = coeffs["tower_coeff"] * q_tower
            p_pump = coeffs["cdu_pump_coeff"] * q_cp_total + coeffs["wshp_pump_coeff"] * q_wshp_in
            p_cdu_free = coeffs["cdu_free_coeff"] * q_cp_free
            p_terminal = (
                sum(float(data["old_air_coeff_eff"][z][t]) * vars_["s_air_old"][z, t].X + coeffs["ac_fan_coeff"] * vars_["s_air_new"][z, t].X for z in zones)
                + coeffs["rdhx_coeff"] * q_rdhx_total
            )
            p_cooling = p_chiller + p_tower + p_ashp + p_wshp + p_pump + p_cdu_free + p_terminal
            q_heat = q_ashp_out + q_wshp_out + vars_["q_tes_dis"][t].X - vars_["q_tes_ch"][t].X
            power_residual = vars_["p_grid"][t].X + vars_["p_bess_dis"][t].X - float(data["it_total"][t]) - p_cooling - vars_["p_bess_ch"][t].X
            heat_residual = q_ashp_out + q_wshp_out + vars_["q_tes_dis"][t].X - q_heat - vars_["q_tes_ch"][t].X
            heat_demand_bound = max(0.0, float(data["heat_demand"][t]))
            heat_demand_upper_violation = max(0.0, q_heat - heat_demand_bound)
            max_heat_demand_upper_violation = max(max_heat_demand_upper_violation, heat_demand_upper_violation)
            max_power_residual = max(max_power_residual, abs(power_residual))
            max_heat_residual = max(max_heat_residual, abs(heat_residual))
            theta_values = []
            for i in rack_range:
                zone = data["rack_to_zone"][i]
                theta_values.append(
                    float(dynamic["supply_temp_by_rack"][i])
                    + float(dynamic["heat_influence"][t, i])
                    - coeffs["beta_air"] * (vars_["s_air_old"][zone, t].X + vars_["s_air_new"][zone, t].X)
                )
            hotspot_slack = (
                sum(vars_["xi"][i, t].X for i in rack_range)
                if bool(coeffs["hotspot_constraints_enabled"]) and "xi" in vars_
                else 0.0
            )
            rows.append(
                {
                    "timestamp_hour_utc": frame.loc[t, "timestamp_hour_utc"],
                    "representative_day_id": frame.loc[t, "representative_day_id"] if "representative_day_id" in frame else 0,
                    "day_weight": float(data["day_weight"][t]),
                    "it_load_kw": float(data["it_total"][t]),
                    "q_air_kw": q_air_total,
                    "q_cp_kw": q_cp_total,
                    "q_rdhx_kw": q_rdhx_total,
                    "s_air_kw": s_air_old_total + s_air_new_total,
                    "s_air_old_kw": s_air_old_total,
                    "s_air_new_kw": s_air_new_total,
                    "old_ac_capacity_eff_kw": sum(float(dynamic["old_air_capacity"][z][t]) for z in zones),
                    "q_air_to_ashp_kw": q_air_to_ashp,
                    "q_air_to_reject_kw": q_air_reject,
                    "q_cp_to_wshp_kw": q_cp_to_wshp,
                    "q_cp_to_free_kw": q_cp_free,
                    "q_rdhx_to_wshp_kw": q_rdhx_to_wshp,
                    "q_rdhx_to_chiller_kw": q_rdhx_chiller,
                    "q_ashp_out_kw": q_ashp_out,
                    "p_ashp_kw": p_ashp,
                    "q_wshp_in_kw": q_wshp_in,
                    "q_wshp_out_kw": q_wshp_out,
                    "p_wshp_kw": p_wshp,
                    "q_chiller_evap_kw": q_chiller_evap,
                    "p_chiller_kw": p_chiller,
                    "q_tower_kw": q_tower,
                    "p_tower_kw": p_tower,
                    "p_pump_kw": p_pump,
                    "p_cdu_free_kw": p_cdu_free,
                    "p_terminal_kw": p_terminal,
                    "p_cooling_kw": p_cooling,
                    "p_grid_kw": vars_["p_grid"][t].X,
                    "q_heat_kw": q_heat,
                    "heating_demand_kw": heat_demand_bound,
                    "heat_demand_upper_violation_kw": heat_demand_upper_violation,
                    "q_tes_ch_kw": vars_["q_tes_ch"][t].X,
                    "q_tes_dis_kw": vars_["q_tes_dis"][t].X,
                    "e_tes_kwh": vars_["e_tes"][t].X,
                    "p_bess_ch_kw": vars_["p_bess_ch"][t].X,
                    "p_bess_dis_kw": vars_["p_bess_dis"][t].X,
                    "e_bess_kwh": vars_["e_bess"][t].X,
                    "max_theta_c": max(theta_values),
                    "hotspot_slack_c": hotspot_slack,
                    "hotspot_penalty_yuan": coeffs["hotspot_penalty"] * float(data["day_weight"][t]) * hotspot_slack,
                    "bess_degradation_cost_yuan": coeffs["bess_deg_cost"] * float(data["day_weight"][t]) * vars_["p_bess_dis"][t].X,
                    "power_balance_residual_kw": power_residual,
                    "heat_balance_residual_kw": heat_residual,
                }
            )

        dispatch = pd.DataFrame(rows)
        diagnostics = {
            "gurobi_status": int(model.Status),
            "objective_value": float(model.ObjVal),
            "rack_count": len(data["rack_ids"]),
            "zone_count": len(zones),
            "max_power_balance_residual_kw": max_power_residual,
            "max_heat_balance_residual_kw": max_heat_residual,
            "max_heat_demand_upper_violation_kw": max_heat_demand_upper_violation,
            "max_hotspot_slack_c": float(dispatch["hotspot_slack_c"].max()),
            "total_hotspot_slack_c_hour": float((dispatch["hotspot_slack_c"] * dispatch["day_weight"]).sum()),
            "max_tes_soc_kwh": float(dispatch["e_tes_kwh"].max()),
            "max_bess_soc_kwh": float(dispatch["e_bess_kwh"].max()),
            "heat_topology_nonzero_edges": int(_heat_adjacency_edge_count(data["heat_adjacency"])),
            "heat_topology_density": float(
                _heat_adjacency_edge_count(data["heat_adjacency"])
                / max(1, len(data["rack_ids"]) * len(data["rack_ids"]))
            ),
            "warm_start_payload_vars": len(self._vars["warm_start"]),
            "inner_extract_s": perf_counter() - extract_started,
            **base_diag,
        }
        return InnerSolveResult(
            True,
            float(model.ObjVal),
            dispatch,
            diagnostics,
            _extract_template_warm_payload(self._vars["warm_start"], zones, times),
        )


def _set_gurobi_params(model: Any, solver: dict[str, Any]) -> None:
    if solver.get("output_flag") is not None:
        model.Params.OutputFlag = int(solver["output_flag"])
    if solver.get("time_limit_seconds"):
        model.Params.TimeLimit = float(solver["time_limit_seconds"])
    if solver.get("mip_gap") is not None:
        model.Params.MIPGap = float(solver["mip_gap"])
    if solver.get("threads") is not None:
        model.Params.Threads = int(solver["threads"])
    if solver.get("mip_focus") is not None:
        model.Params.MIPFocus = int(solver["mip_focus"])
    if solver.get("nodefile_start_gb") is not None:
        model.Params.NodefileStart = float(solver["nodefile_start_gb"])
    if solver.get("numeric_focus") is not None:
        model.Params.NumericFocus = int(solver["numeric_focus"])


def _extract_template_warm_payload(variable_groups: dict[str, Any], zones: list[str], times: list[int]) -> dict[str, Any]:
    from inner_dispatch import _extract_warm_start_payload

    return _extract_warm_start_payload(variable_groups, zones, times)
