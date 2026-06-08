from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from decision import PlanningDecision


@dataclass(frozen=True)
class InnerSolveResult:
    feasible: bool
    objective_value: float
    dispatch: pd.DataFrame
    diagnostics: dict[str, Any]
    warm_start_payload: dict[str, Any] = field(default_factory=dict)


class InnerDispatchMILP:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

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
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for inner MILP dispatch.") from exc

        frame = time_frame.reset_index(drop=True).copy()
        if frame.empty:
            return InnerSolveResult(
                False,
                float("inf"),
                pd.DataFrame(),
                {"reason": "empty_time_frame", "build_mode": build_mode or "fresh"},
            )

        tech = self.config.get("technology", {})
        fitted = tech.get("fitted", {})
        solver = _gurobi_solver_config(self.config)
        scenario = self.config.get("scenario", {})

        cop_chiller = _positive_float(fitted.get("cop_chiller", tech.get("cop_chiller", 3.0)), 3.0)
        cop_ashp = _positive_float(tech.get("cop_ashp", 3.2), 3.2)
        cop_wshp = _positive_float(tech.get("cop_wshp", 4.0), 4.0)
        ac_fan_coeff = max(0.0, float(fitted.get("ac_fan_kw_per_kw", tech.get("ac_fan_kw_per_kw", 0.2137))))
        rdhx_coeff = max(0.0, float(fitted.get("e_rdhx_kw_per_kw", tech.get("e_rdhx_kw_per_kw", 0.0265))))
        tower_coeff = max(0.0, float(tech.get("tower_kw_per_kw_heat", 0.015)))
        cdu_free_coeff = max(0.0, float(tech.get("cdu_free_kw_per_kw", tech.get("cdu_pump_kw_per_kw", 0.02))))
        cdu_pump_coeff = max(0.0, float(tech.get("cdu_pump_kw_per_kw", 0.02)))
        wshp_pump_coeff = max(0.0, float(tech.get("wshp_pump_kw_per_kw", 0.01)))
        heat_credit = float(self.config.get("economics", {}).get("heat_credit_yuan_per_kwh", 0.0))
        hotspot_penalty = float(scenario.get("thermal_slack_penalty_yuan_per_c_hour", scenario.get("thermal_slack_penalty", 1.0e6)))
        bess_deg_cost = float(tech.get("bess_degradation_yuan_per_kwh", 0.0))

        liquid = tech.get("liquid_heat_fraction", {})
        cp_fraction = float(liquid.get("cold_plate", tech.get("cold_plate_heat_fraction", 0.7)))
        rdhx_fraction = float(liquid.get("rdhx", 1.0))
        cp_fraction = min(max(cp_fraction, 0.0), 1.0)
        rdhx_fraction = min(max(rdhx_fraction, 0.0), 1.0)

        thermal = tech.get("thermal", {})
        hotspot_constraints_enabled = _hotspot_constraints_enabled(self.config)
        old_supply_temp_c = float(thermal.get("old_supply_temp_c", 24.0))
        new_supply_temp_c = float(thermal.get("new_supply_temp_c", 22.0))
        aux_supply_temp_c = float(thermal.get("aux_supply_temp_c", 23.0))
        rack_temp_max_c = float(thermal.get("rack_temp_max_c", 27.0))
        beta_air = float(thermal.get("beta_air_c_per_kw", 0.02))

        tes = tech.get("tes", {})
        tes_loss = min(max(float(tes.get("loss_per_hour", tech.get("tes_loss_per_hour", 0.002))), 0.0), 1.0)
        tes_eta_ch = min(max(float(tes.get("charge_efficiency", 0.95)), 1.0e-6), 1.0)
        tes_eta_dis = min(max(float(tes.get("discharge_efficiency", 0.95)), 1.0e-6), 1.0)
        tes_power_to_energy = max(0.0, float(tes.get("power_to_energy", 0.5)))

        bess = tech.get("bess", {})
        rte = min(max(float(bess.get("roundtrip_efficiency", tech.get("bess_roundtrip_efficiency", 0.9))), 1.0e-6), 1.0)
        bess_eta_ch = min(max(float(bess.get("charge_efficiency", np.sqrt(rte))), 1.0e-6), 1.0)
        bess_eta_dis = min(max(float(bess.get("discharge_efficiency", np.sqrt(rte))), 1.0e-6), 1.0)
        bess_loss = min(max(float(bess.get("self_discharge_per_hour", 0.0)), 0.0), 1.0)
        bess_soc_min = min(max(float(bess.get("soc_min", 0.1)), 0.0), 1.0)
        bess_soc_max = min(max(float(bess.get("soc_max", 0.9)), bess_soc_min), 1.0)
        bess_c_rate = max(0.0, float(bess.get("c_rate_per_hour", 0.5)))

        rack_cols = _rack_load_columns(frame)
        rack_ids = [col.removesuffix("_kw") for col in rack_cols]
        if not rack_cols:
            rack_ids, rack_cols, frame = _add_synthetic_racks(frame, decision.zone_ids)

        rack_to_zone = _rack_zone_map(self.config, rack_ids, decision.zone_ids)
        heat_matrix = _heat_matrix(self.config, len(rack_ids))
        heat_adjacency = _heat_adjacency(heat_matrix, len(rack_ids))
        times = list(range(len(frame)))
        zones = list(decision.zone_ids)
        day_weight = frame.get("day_weight", pd.Series(1.0, index=frame.index)).astype(float).to_numpy()
        price = frame.get("price_yuan_per_kwh", pd.Series(0.0, index=frame.index)).astype(float).to_numpy()
        heat_demand = frame.get("heating_demand_kw", pd.Series(0.0, index=frame.index)).astype(float).to_numpy()
        it_total = frame.get("it_load_kw", frame[rack_cols].sum(axis=1)).astype(float).to_numpy()

        q_it = frame[rack_cols].astype(float).to_numpy()
        q_cp = np.zeros_like(q_it)
        q_rdhx = np.zeros_like(q_it)
        q_air = np.zeros_like(q_it)
        for ridx, zone_id in enumerate(rack_to_zone):
            raw = np.maximum(q_it[:, ridx], 0.0)
            cp = cp_fraction * raw if decision.p_z.get(zone_id, 0) else np.zeros_like(raw)
            rdhx = rdhx_fraction * raw if decision.r_z.get(zone_id, 0) else np.zeros_like(raw)
            liquid_total = np.minimum(raw, cp + rdhx)
            scale = np.divide(raw, cp + rdhx, out=np.ones_like(raw), where=(cp + rdhx) > raw)
            q_cp[:, ridx] = cp * scale
            q_rdhx[:, ridx] = rdhx * scale
            q_air[:, ridx] = np.maximum(0.0, raw - q_cp[:, ridx] - q_rdhx[:, ridx])

        rack_count = len(rack_ids)
        time_count = len(times)
        rack_range = range(rack_count)
        racks_by_zone = {
            zone: np.asarray([idx for idx, z in enumerate(rack_to_zone) if z == zone], dtype=int)
            for zone in zones
        }
        q_air_zone = {
            zone: _sum_columns(q_air, racks_by_zone[zone], time_count)
            for zone in zones
        }
        q_cp_zone = {
            zone: _sum_columns(q_cp, racks_by_zone[zone], time_count)
            for zone in zones
        }
        q_rdhx_zone = {
            zone: _sum_columns(q_rdhx, racks_by_zone[zone], time_count)
            for zone in zones
        }
        old_air_capacity_eff = {
            zone: _time_series_array(frame, f"old_ac_capacity_eff_{zone}_kw", peak_load_by_zone_kw.get(zone, 0.0))
            for zone in zones
        }
        old_air_coeff_eff = {
            zone: _time_series_array(frame, f"old_ac_terminal_coeff_{zone}_kw_per_kw", ac_fan_coeff)
            for zone in zones
        }
        psi_amb = {
            zone: _time_series_array(frame, f"psi_amb_{zone}", 1.0)
            for zone in zones
        }
        old_air_capacity = {
            zone: (
                np.maximum(0.0, old_air_capacity_eff[zone] * psi_amb[zone])
                if decision.o_z.get(zone, 0)
                else np.zeros(time_count, dtype=float)
            )
            for zone in zones
        }
        new_air_capacity = {zone: float(decision.cap_ac_new[zone]) for zone in zones}
        air_capacity = {
            zone: old_air_capacity[zone] + new_air_capacity[zone]
            for zone in zones
        }
        a_flag = {zone: int(decision.a_z.get(zone, 0)) for zone in zones}
        cp_total = np.sum([q_cp_zone[zone] for zone in zones], axis=0) if zones else np.zeros(time_count)
        rdhx_total = np.sum([q_rdhx_zone[zone] for zone in zones], axis=0) if zones else np.zeros(time_count)
        heat_influence = _heat_influence_from_adjacency(q_air, heat_adjacency)
        supply_temp_by_rack = np.asarray(
            [
                _supply_temp(decision, zone_id, old_supply_temp_c, new_supply_temp_c, aux_supply_temp_c)
                for zone_id in rack_to_zone
            ],
            dtype=float,
        )

        model = gp.Model("new_research_inner_dispatch")
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

        zt = [(z, t) for z in zones for t in times]
        s_air_old = model.addVars(zt, lb=0.0, name="s_air_old")
        s_air_new = model.addVars(zt, lb=0.0, name="s_air_new")
        q_air_to_ashp = model.addVars(zt, lb=0.0, name="q_air_to_ashp")
        q_cp_to_wshp = model.addVars(zt, lb=0.0, name="q_cp_to_wshp")
        q_rdhx_to_wshp = model.addVars(zt, lb=0.0, name="q_rdhx_to_wshp")
        p_grid = model.addVars(times, lb=0.0, name="p_grid")

        tes_power_ub = max(0.0, decision.cap_tes * tes_power_to_energy)
        q_tes_ch = model.addVars(times, lb=0.0, ub=tes_power_ub, name="q_tes_ch")
        q_tes_dis = model.addVars(times, lb=0.0, ub=tes_power_ub, name="q_tes_dis")
        e_tes = model.addVars(times, lb=0.0, ub=max(0.0, decision.cap_tes), name="e_tes")
        d_tes_ch = model.addVars(times, vtype=GRB.BINARY, name="d_tes_ch")
        d_tes_dis = model.addVars(times, vtype=GRB.BINARY, name="d_tes_dis")

        bess_charge_limit_kw = max(0.0, decision.cap_bess * bess_c_rate)
        p_bess_ch = model.addVars(times, lb=0.0, ub=bess_charge_limit_kw, name="p_bess_ch")
        p_bess_dis = model.addVars(times, lb=0.0, ub=bess_charge_limit_kw, name="p_bess_dis")
        e_bess = model.addVars(
            times,
            lb=max(0.0, decision.cap_bess * bess_soc_min),
            ub=max(0.0, decision.cap_bess * bess_soc_max),
            name="e_bess",
        )
        d_bess_ch = model.addVars(times, vtype=GRB.BINARY, name="d_bess_ch")
        d_bess_dis = model.addVars(times, vtype=GRB.BINARY, name="d_bess_dis")

        xi = model.addVars(rack_range, times, lb=0.0, name="xi") if hotspot_constraints_enabled else None
        warm_start_vars = {
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

        s_air_expr = {(z, t): s_air_old[z, t] + s_air_new[z, t] for z, t in zt}
        q_air_to_reject_expr = {
            (z, t): float(q_air_zone[z][t]) - q_air_to_ashp[z, t]
            for z, t in zt
        }
        q_cp_to_free_expr = {
            (z, t): float(q_cp_zone[z][t]) - q_cp_to_wshp[z, t]
            for z, t in zt
        }
        q_rdhx_to_chiller_expr = {
            (z, t): float(q_rdhx_zone[z][t]) - q_rdhx_to_wshp[z, t]
            for z, t in zt
        }
        model.addConstrs(
            (q_air_to_ashp[z, t] <= a_flag[z] * float(q_air_zone[z][t]) for z, t in zt),
            name="air_ashp_enabled",
        )
        model.addConstrs(
            (q_air_to_reject_expr[z, t] <= float(air_capacity[z][t]) for z, t in zt),
            name="air_reject_capacity",
        )
        model.addConstrs(
            (s_air_old[z, t] <= float(old_air_capacity[z][t]) for z, t in zt),
            name="old_air_capacity",
        )
        model.addConstrs(
            (s_air_new[z, t] <= new_air_capacity[z] for z, t in zt),
            name="new_air_capacity",
        )
        model.addConstrs(
            (s_air_expr[z, t] >= float(q_air_zone[z][t]) for z, t in zt),
            name="air_service_min",
        )
        model.addConstrs(
            (s_air_expr[z, t] <= float(air_capacity[z][t]) for z, t in zt),
            name="air_service_capacity",
        )
        model.addConstrs(
            (q_cp_to_wshp[z, t] <= float(q_cp_zone[z][t]) for z, t in zt),
            name="cp_wshp_upper",
        )
        model.addConstrs(
            (0.0 * q_cp_to_wshp[z, t] + float(q_cp_zone[z][t]) <= float(decision.cap_cdu[z]) + 1.0e-7 for z, t in zt),
            name="cdu_capacity",
        )
        model.addConstrs(
            (q_rdhx_to_wshp[z, t] <= float(q_rdhx_zone[z][t]) for z, t in zt),
            name="rdhx_wshp_upper",
        )
        model.addConstrs(
            (0.0 * q_rdhx_to_wshp[z, t] + float(q_rdhx_zone[z][t]) <= float(decision.cap_rdhx[z]) + 1.0e-7 for z, t in zt),
            name="rdhx_capacity",
        )

        ashp_source_to_output = cop_ashp / max(cop_ashp - 1.0, 1.0e-9)
        wshp_source_to_output = cop_wshp / max(cop_wshp - 1.0, 1.0e-9)
        q_air_to_ashp_sum = {
            t: gp.quicksum(q_air_to_ashp[z, t] for z in zones)
            for t in times
        }
        q_cp_free_sum = {
            t: gp.quicksum(q_cp_to_free_expr[z, t] for z in zones)
            for t in times
        }
        q_cp_wshp_sum = {
            t: gp.quicksum(q_cp_to_wshp[z, t] for z in zones)
            for t in times
        }
        q_rdhx_wshp_sum = {
            t: gp.quicksum(q_rdhx_to_wshp[z, t] for z in zones)
            for t in times
        }
        q_air_reject_sum = {
            t: gp.quicksum(q_air_to_reject_expr[z, t] for z in zones)
            for t in times
        }
        q_rdhx_chiller_sum = {
            t: gp.quicksum(q_rdhx_to_chiller_expr[z, t] for z in zones)
            for t in times
        }
        p_terminal_old_new = {
            t: gp.quicksum(
                float(old_air_coeff_eff[z][t]) * s_air_old[z, t] + ac_fan_coeff * s_air_new[z, t]
                for z in zones
            )
            for t in times
        }
        q_ashp_in_expr = q_air_to_ashp_sum
        q_ashp_out_expr = {t: ashp_source_to_output * q_ashp_in_expr[t] for t in times}
        p_ashp_expr = {t: q_ashp_out_expr[t] / cop_ashp for t in times}
        q_wshp_in_expr = {
            t: q_cp_wshp_sum[t] + q_rdhx_wshp_sum[t]
            for t in times
        }
        q_wshp_out_expr = {t: wshp_source_to_output * q_wshp_in_expr[t] for t in times}
        p_wshp_expr = {t: q_wshp_out_expr[t] / cop_wshp for t in times}
        q_chiller_evap_expr = {
            t: q_air_reject_sum[t] + q_rdhx_chiller_sum[t]
            for t in times
        }
        p_chiller_expr = {t: q_chiller_evap_expr[t] / cop_chiller for t in times}
        q_tower_expr = {
            t: q_chiller_evap_expr[t] + p_chiller_expr[t] + q_cp_free_sum[t]
            for t in times
        }
        p_tower_expr = {t: tower_coeff * q_tower_expr[t] for t in times}
        p_pump_expr = {
            t: cdu_pump_coeff * float(cp_total[t]) + wshp_pump_coeff * q_wshp_in_expr[t]
            for t in times
        }
        p_cdu_free_expr = {t: cdu_free_coeff * q_cp_free_sum[t] for t in times}
        p_terminal_expr = {
            t: p_terminal_old_new[t] + rdhx_coeff * float(rdhx_total[t])
            for t in times
        }
        q_heat_expr = {
            t: q_ashp_out_expr[t] + q_wshp_out_expr[t] + q_tes_dis[t] - q_tes_ch[t]
            for t in times
        }
        model.addConstrs((q_ashp_out_expr[t] <= max(0.0, decision.cap_ashp) for t in times), name="ashp_capacity")
        model.addConstrs(
            (q_wshp_out_expr[t] <= max(0.0, decision.cap_wshp) for t in times),
            name="wshp_capacity",
        )
        model.addConstrs(
            (q_chiller_evap_expr[t] <= max(0.0, chiller_old_kw + decision.delta_cap_chiller) for t in times),
            name="chiller_capacity",
        )
        model.addConstrs(
            (q_tower_expr[t] <= max(0.0, tower_old_kw + decision.delta_cap_tower) for t in times),
            name="tower_capacity",
        )
        model.addConstrs(
            (q_heat_expr[t] >= 0.0 for t in times),
            name="heat_supply_nonnegative",
        )
        model.addConstrs((q_heat_expr[t] <= max(0.0, float(heat_demand[t])) for t in times), name="heat_demand")
        model.addConstrs(
            (
                p_grid[t] + p_bess_dis[t]
                == float(it_total[t])
                + p_chiller_expr[t]
                + p_tower_expr[t]
                + p_ashp_expr[t]
                + p_wshp_expr[t]
                + p_pump_expr[t]
                + p_cdu_free_expr[t]
                + p_terminal_expr[t]
                + p_bess_ch[t]
                for t in times
            ),
            name="power_balance",
        )
        model.addConstrs((q_tes_ch[t] <= tes_power_ub * d_tes_ch[t] for t in times), name="tes_charge_limit")
        model.addConstrs((q_tes_dis[t] <= tes_power_ub * d_tes_dis[t] for t in times), name="tes_discharge_limit")
        model.addConstrs((d_tes_ch[t] + d_tes_dis[t] <= 1 for t in times), name="tes_charge_discharge_mutex")
        model.addConstrs((p_bess_ch[t] <= bess_charge_limit_kw * d_bess_ch[t] for t in times), name="bess_charge_limit")
        model.addConstrs((p_bess_dis[t] <= bess_charge_limit_kw * d_bess_dis[t] for t in times), name="bess_discharge_limit")
        model.addConstrs((d_bess_ch[t] + d_bess_dis[t] <= 1 for t in times), name="bess_charge_discharge_mutex")

        for day_indices in _cyclic_groups(frame):
            for pos, t in enumerate(day_indices):
                prev = day_indices[pos - 1]
                model.addConstr(e_tes[t] == (1.0 - tes_loss) * e_tes[prev] + tes_eta_ch * q_tes_ch[t] - q_tes_dis[t] / tes_eta_dis, name=f"tes_soc[{t}]")
                model.addConstr(e_bess[t] == (1.0 - bess_loss) * e_bess[prev] + bess_eta_ch * p_bess_ch[t] - p_bess_dis[t] / bess_eta_dis, name=f"bess_soc[{t}]")

        theta_expr = {
            (i, t): (
                float(supply_temp_by_rack[i])
                + float(heat_influence[t, i])
                - beta_air * s_air_expr[rack_to_zone[i], t]
            )
            for i in rack_range
            for t in times
        }
        if hotspot_constraints_enabled and xi is not None:
            model.addConstrs(
                (theta_expr[i, t] <= rack_temp_max_c + xi[i, t] for i in rack_range for t in times),
                name="hotspot_limit",
            )

        model.setObjective(
            gp.quicksum(
                float(day_weight[t])
                * (
                    float(price[t]) * p_grid[t]
                    - heat_credit * q_heat_expr[t]
                    + bess_deg_cost * p_bess_dis[t]
                    + (
                        hotspot_penalty * gp.quicksum(xi[i, t] for i in rack_range)
                        if hotspot_constraints_enabled and xi is not None
                        else 0.0
                    )
                )
                for t in times
            ),
            GRB.MINIMIZE,
        )
        warm_start_values_applied = _apply_warm_start(warm_start_vars, warm_start, zones, times)
        optimize_started = perf_counter()
        model.optimize()
        optimize_finished = perf_counter()

        if model.SolCount <= 0:
            return InnerSolveResult(
                feasible=False,
                objective_value=float("inf"),
                dispatch=pd.DataFrame(),
                diagnostics={
                    "gurobi_status": int(model.Status),
                    "reason": "no_solution",
                    **_gurobi_diagnostics(solver),
                    "model_sections": _model_sections(),
                    "build_mode": build_mode or "fresh",
                    "warm_start_attempted": bool(warm_start),
                    "warm_start_values_applied": warm_start_values_applied,
                    "hotspot_constraints_enabled": hotspot_constraints_enabled,
                    "inner_build_total_s": optimize_started - solve_started,
                    "inner_update_s": 0.0,
                    "inner_optimize_s": optimize_finished - optimize_started,
                    "template_build_once_s": 0.0,
                    "template_reuse_hit": False,
                },
            )

        extract_started = perf_counter()
        rows: list[dict[str, Any]] = []
        max_power_residual = 0.0
        max_heat_residual = 0.0
        max_heat_demand_upper_violation = 0.0
        for t in times:
            q_air_total = sum(float(q_air_zone[z][t]) for z in zones)
            q_cp_total = sum(float(q_cp_zone[z][t]) for z in zones)
            q_rdhx_total = sum(float(q_rdhx_zone[z][t]) for z in zones)
            s_air_total = sum(_solution_value(s_air_expr[z, t]) for z in zones)
            s_air_old_total = sum(s_air_old[z, t].X for z in zones)
            s_air_new_total = sum(s_air_new[z, t].X for z in zones)
            old_air_capacity_total = sum(float(old_air_capacity[z][t]) for z in zones)
            hotspot_slack = (
                sum(xi[i, t].X for i in rack_range)
                if hotspot_constraints_enabled and xi is not None
                else 0.0
            )
            p_cooling = (
                _solution_value(p_chiller_expr[t])
                + _solution_value(p_tower_expr[t])
                + _solution_value(p_ashp_expr[t])
                + _solution_value(p_wshp_expr[t])
                + _solution_value(p_pump_expr[t])
                + _solution_value(p_cdu_free_expr[t])
                + _solution_value(p_terminal_expr[t])
            )
            power_residual = p_grid[t].X + p_bess_dis[t].X - float(it_total[t]) - p_cooling - p_bess_ch[t].X
            heat_residual = (
                _solution_value(q_ashp_out_expr[t])
                + _solution_value(q_wshp_out_expr[t])
                + q_tes_dis[t].X
                - _solution_value(q_heat_expr[t])
                - q_tes_ch[t].X
            )
            heat_supply = _solution_value(q_heat_expr[t])
            heat_demand_upper_violation = max(0.0, heat_supply - max(0.0, float(heat_demand[t])))
            max_heat_demand_upper_violation = max(max_heat_demand_upper_violation, heat_demand_upper_violation)
            max_power_residual = max(max_power_residual, abs(power_residual))
            max_heat_residual = max(max_heat_residual, abs(heat_residual))
            rows.append(
                {
                    "timestamp_hour_utc": frame.loc[t, "timestamp_hour_utc"],
                    "representative_day_id": frame.loc[t, "representative_day_id"] if "representative_day_id" in frame else 0,
                    "day_weight": float(day_weight[t]),
                    "it_load_kw": float(it_total[t]),
                    "q_air_kw": q_air_total,
                    "q_cp_kw": q_cp_total,
                    "q_rdhx_kw": q_rdhx_total,
                    "s_air_kw": s_air_total,
                    "s_air_old_kw": s_air_old_total,
                    "s_air_new_kw": s_air_new_total,
                    "old_ac_capacity_eff_kw": old_air_capacity_total,
                    "q_air_to_ashp_kw": sum(q_air_to_ashp[z, t].X for z in zones),
                    "q_air_to_reject_kw": sum(_solution_value(q_air_to_reject_expr[z, t]) for z in zones),
                    "q_cp_to_wshp_kw": sum(q_cp_to_wshp[z, t].X for z in zones),
                    "q_cp_to_free_kw": sum(_solution_value(q_cp_to_free_expr[z, t]) for z in zones),
                    "q_rdhx_to_wshp_kw": sum(q_rdhx_to_wshp[z, t].X for z in zones),
                    "q_rdhx_to_chiller_kw": sum(_solution_value(q_rdhx_to_chiller_expr[z, t]) for z in zones),
                    "q_ashp_out_kw": _solution_value(q_ashp_out_expr[t]),
                    "p_ashp_kw": _solution_value(p_ashp_expr[t]),
                    "q_wshp_in_kw": _solution_value(q_wshp_in_expr[t]),
                    "q_wshp_out_kw": _solution_value(q_wshp_out_expr[t]),
                    "p_wshp_kw": _solution_value(p_wshp_expr[t]),
                    "q_chiller_evap_kw": _solution_value(q_chiller_evap_expr[t]),
                    "p_chiller_kw": _solution_value(p_chiller_expr[t]),
                    "q_tower_kw": _solution_value(q_tower_expr[t]),
                    "p_tower_kw": _solution_value(p_tower_expr[t]),
                    "p_pump_kw": _solution_value(p_pump_expr[t]),
                    "p_cdu_free_kw": _solution_value(p_cdu_free_expr[t]),
                    "p_terminal_kw": _solution_value(p_terminal_expr[t]),
                    "p_cooling_kw": p_cooling,
                    "p_grid_kw": p_grid[t].X,
                    "q_heat_kw": heat_supply,
                    "heating_demand_kw": max(0.0, float(heat_demand[t])),
                    "heat_demand_upper_violation_kw": heat_demand_upper_violation,
                    "q_tes_ch_kw": q_tes_ch[t].X,
                    "q_tes_dis_kw": q_tes_dis[t].X,
                    "e_tes_kwh": e_tes[t].X,
                    "p_bess_ch_kw": p_bess_ch[t].X,
                    "p_bess_dis_kw": p_bess_dis[t].X,
                    "e_bess_kwh": e_bess[t].X,
                    "max_theta_c": max(_solution_value(theta_expr[i, t]) for i in rack_range),
                    "hotspot_slack_c": hotspot_slack,
                    "hotspot_penalty_yuan": hotspot_penalty * float(day_weight[t]) * hotspot_slack,
                    "bess_degradation_cost_yuan": bess_deg_cost * float(day_weight[t]) * p_bess_dis[t].X,
                    "power_balance_residual_kw": power_residual,
                    "heat_balance_residual_kw": heat_residual,
                }
            )

        dispatch = pd.DataFrame(rows)
        diagnostics = {
            "gurobi_status": int(model.Status),
            "objective_value": float(model.ObjVal),
            "rack_count": len(rack_ids),
            "zone_count": len(zones),
            "max_power_balance_residual_kw": max_power_residual,
            "max_heat_balance_residual_kw": max_heat_residual,
            "max_heat_demand_upper_violation_kw": max_heat_demand_upper_violation,
            "hotspot_constraints_enabled": hotspot_constraints_enabled,
            "max_hotspot_slack_c": float(dispatch["hotspot_slack_c"].max()),
            "total_hotspot_slack_c_hour": float((dispatch["hotspot_slack_c"] * dispatch["day_weight"]).sum()),
            "max_tes_soc_kwh": float(dispatch["e_tes_kwh"].max()),
            "max_bess_soc_kwh": float(dispatch["e_bess_kwh"].max()),
            "heat_topology_nonzero_edges": int(_heat_adjacency_edge_count(heat_adjacency)),
            "heat_topology_density": float(
                _heat_adjacency_edge_count(heat_adjacency) / max(1, rack_count * rack_count)
            ),
            **_gurobi_diagnostics(solver),
            "model_sections": _model_sections(),
            "build_mode": build_mode or "fresh",
            "warm_start_attempted": bool(warm_start),
            "warm_start_values_applied": warm_start_values_applied,
            "warm_start_payload_vars": len(warm_start_vars),
            "inner_build_total_s": optimize_started - solve_started,
            "inner_update_s": 0.0,
            "inner_optimize_s": optimize_finished - optimize_started,
            "inner_extract_s": perf_counter() - extract_started,
            "template_build_once_s": 0.0,
            "template_reuse_hit": False,
        }
        return InnerSolveResult(
            True,
            float(model.ObjVal),
            dispatch,
            diagnostics,
            _extract_warm_start_payload(warm_start_vars, zones, times),
        )


def _apply_warm_start(
    variable_groups: dict[str, Any],
    payload: dict[str, Any] | None,
    zone_ids: list[str] | None = None,
    times: list[int] | None = None,
) -> int:
    try:
        from gurobipy import GRB
        undefined = GRB.UNDEFINED
    except Exception:
        undefined = 1.0e101

    for variables in variable_groups.values():
        for variable in variables.values():
            try:
                variable.Start = undefined
            except Exception:
                continue

    if not payload:
        return 0

    if payload.get("__format__") == "dense_v1":
        return _apply_dense_warm_start(variable_groups, payload, zone_ids, times)

    applied = 0
    for group_name, values in payload.items():
        variables = variable_groups.get(group_name)
        if variables is None or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key not in variables:
                continue
            try:
                variables[key].Start = float(value)
            except Exception:
                continue
            applied += 1
    return applied


def _apply_dense_warm_start(
    variable_groups: dict[str, Any],
    payload: dict[str, Any],
    zone_ids: list[str] | None,
    times: list[int] | None,
) -> int:
    vars_payload = payload.get("vars", {})
    if not isinstance(vars_payload, dict):
        return 0
    payload_zones = [str(zone) for zone in payload.get("zone_ids", zone_ids or [])]
    payload_times = list(range(int(payload.get("time_count", len(times or [])))))
    current_zone_set = set(zone_ids or [])
    current_time_set = set(times or [])
    applied = 0
    for group_name, values in vars_payload.items():
        variables = variable_groups.get(group_name)
        if variables is None:
            continue
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            if not payload_zones or not current_zone_set:
                continue
            for zidx, zone in enumerate(payload_zones):
                if zone not in current_zone_set or zidx >= arr.shape[0]:
                    continue
                width = min(arr.shape[1], len(payload_times))
                for tidx in range(width):
                    t = payload_times[tidx]
                    if t not in current_time_set or (zone, t) not in variables:
                        continue
                    try:
                        variables[zone, t].Start = float(arr[zidx, tidx])
                    except Exception:
                        continue
                    applied += 1
        elif arr.ndim == 1:
            width = min(arr.shape[0], len(payload_times))
            for tidx in range(width):
                t = payload_times[tidx]
                if t not in current_time_set or t not in variables:
                    continue
                try:
                    variables[t].Start = float(arr[tidx])
                except Exception:
                    continue
                applied += 1
    return applied


def _extract_warm_start_payload(
    variable_groups: dict[str, Any],
    zone_ids: list[str] | None = None,
    times: list[int] | None = None,
) -> dict[str, Any]:
    zones = list(zone_ids or [])
    time_list = list(times or [])
    if zones and time_list:
        vars_payload: dict[str, np.ndarray] = {}
        zone_index = {zone: idx for idx, zone in enumerate(zones)}
        for group_name, variables in variable_groups.items():
            sample_key = next(iter(variables.keys()), None)
            if isinstance(sample_key, tuple) and len(sample_key) == 2:
                arr = np.zeros((len(zones), len(time_list)), dtype=float)
                time_index = {t: idx for idx, t in enumerate(time_list)}
                found = False
                for key, variable in variables.items():
                    zone, t = key
                    if zone not in zone_index or t not in time_index:
                        continue
                    try:
                        arr[zone_index[zone], time_index[t]] = float(variable.X)
                    except Exception:
                        continue
                    found = True
                if found:
                    vars_payload[group_name] = arr.copy()
            else:
                arr = np.zeros(len(time_list), dtype=float)
                time_index = {t: idx for idx, t in enumerate(time_list)}
                found = False
                for key, variable in variables.items():
                    if key not in time_index:
                        continue
                    try:
                        arr[time_index[key]] = float(variable.X)
                    except Exception:
                        continue
                    found = True
                if found:
                    vars_payload[group_name] = arr.copy()
        return {
            "__format__": "dense_v1",
            "zone_ids": list(zones),
            "time_count": len(time_list),
            "vars": vars_payload,
        }

    payload: dict[str, dict[Any, float]] = {}
    for group_name, variables in variable_groups.items():
        values: dict[Any, float] = {}
        for key, variable in variables.items():
            try:
                values[key] = float(variable.X)
            except Exception:
                continue
        if values:
            payload[group_name] = values
    return payload


def _solution_value(expr: Any) -> float:
    if isinstance(expr, (int, float, np.integer, np.floating)):
        return float(expr)
    try:
        return float(expr.getValue())
    except AttributeError:
        return float(expr.X)


def _gurobi_solver_config(config: dict[str, Any]) -> dict[str, Any]:
    solver = config.get("solver", {})
    gurobi = solver.get("gurobi", {}) if isinstance(solver.get("gurobi", {}), dict) else {}
    return {
        "time_limit_seconds": gurobi.get(
            "time_limit_seconds",
            gurobi.get("time_limit", solver.get("time_limit_seconds", solver.get("time_limit"))),
        ),
        "mip_gap": gurobi.get("mip_gap", solver.get("mip_gap")),
        "threads": gurobi.get("threads", solver.get("threads")),
        "mip_focus": gurobi.get("mip_focus", solver.get("mip_focus")),
        "nodefile_start_gb": gurobi.get("nodefile_start_gb", solver.get("nodefile_start_gb")),
        "numeric_focus": gurobi.get("numeric_focus", solver.get("numeric_focus")),
        "output_flag": gurobi.get("output_flag", solver.get("output_flag", 0)),
    }


def _hotspot_constraints_enabled(config: dict[str, Any]) -> bool:
    tech = config.get("technology", {})
    thermal = tech.get("thermal", {}) if isinstance(tech, dict) else {}
    if isinstance(thermal, dict) and "hotspot_constraints_enabled" in thermal:
        return bool(thermal["hotspot_constraints_enabled"])
    scenario = config.get("scenario", {})
    if isinstance(scenario, dict) and "hotspot_constraints_enabled" in scenario:
        return bool(scenario["hotspot_constraints_enabled"])
    return True


def _gurobi_diagnostics(solver: dict[str, Any]) -> dict[str, Any]:
    return {
        "gurobi_time_limit_seconds": solver.get("time_limit_seconds"),
        "gurobi_mip_gap": solver.get("mip_gap"),
        "gurobi_threads": solver.get("threads"),
        "gurobi_mip_focus": solver.get("mip_focus"),
        "gurobi_nodefile_start_gb": solver.get("nodefile_start_gb"),
        "gurobi_numeric_focus": solver.get("numeric_focus"),
        "gurobi_output_flag": solver.get("output_flag"),
    }


def _model_sections() -> list[str]:
    return [
        "sets_and_data",
        "heat_split",
        "variables",
        "zone_heat_flow",
        "equipment",
        "storage",
        "electric_and_heat_balance",
        "hotspot",
        "objective",
        "diagnostics",
    ]


def _positive_float(value: Any, default: float) -> float:
    number = float(value)
    return number if number > 0.0 else default


def _time_series_value(frame: pd.DataFrame, col: str, index: int, default: float) -> float:
    if col not in frame.columns:
        return float(default)
    value = pd.to_numeric(pd.Series([frame.loc[index, col]]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(float(value)):
        return float(default)
    return float(value)


def _time_series_array(frame: pd.DataFrame, col: str, default: float) -> np.ndarray:
    if col not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, float(default)).astype(float, copy=False)


def _sum_columns(values: np.ndarray, column_indices: np.ndarray, time_count: int) -> np.ndarray:
    if column_indices.size == 0:
        return np.zeros(time_count, dtype=float)
    return np.asarray(values[:, column_indices].sum(axis=1), dtype=float)


def _heat_adjacency(
    heat_matrix: np.ndarray,
    rack_count: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    count = int(rack_count or 0)
    if heat_matrix.size == 0:
        return tuple((np.asarray([], dtype=int), np.asarray([], dtype=float)) for _ in range(count))

    matrix = np.nan_to_num(np.asarray(heat_matrix, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    count = int(rack_count or matrix.shape[0])
    if matrix.shape != (count, count):
        return tuple((np.asarray([], dtype=int), np.asarray([], dtype=float)) for _ in range(count))

    adjacency: list[tuple[np.ndarray, np.ndarray]] = []
    for target in range(count):
        source_idx = np.flatnonzero(matrix[:, target])
        adjacency.append((source_idx.astype(int, copy=False), matrix[source_idx, target].astype(float, copy=True)))
    return tuple(adjacency)


def _heat_adjacency_edge_count(adjacency: tuple[tuple[np.ndarray, np.ndarray], ...]) -> int:
    return int(sum(len(source_idx) for source_idx, _coeffs in adjacency))


def _heat_influence_from_adjacency(
    q_air_kw: np.ndarray,
    adjacency: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    if q_air_kw.size == 0 or not adjacency:
        return np.zeros_like(q_air_kw, dtype=float)
    if len(adjacency) != q_air_kw.shape[1]:
        return np.zeros_like(q_air_kw, dtype=float)

    out = np.zeros((q_air_kw.shape[0], len(adjacency)), dtype=float)
    for target, (source_idx, coeffs) in enumerate(adjacency):
        if len(source_idx) == 0:
            continue
        out[:, target] = q_air_kw[:, source_idx] @ coeffs
    return out


def _heat_influence_matrix(q_air_kw: np.ndarray, heat_matrix: np.ndarray) -> np.ndarray:
    return _heat_influence_from_adjacency(q_air_kw, _heat_adjacency(heat_matrix, q_air_kw.shape[1]))


def _rack_load_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if "rack_it_" in col and col.endswith("_kw")]


def _add_synthetic_racks(frame: pd.DataFrame, zone_ids: list[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    out = frame.copy()
    zones = zone_ids or ["zone_1"]
    load = out["it_load_kw"].astype(float) / max(1, len(zones))
    rack_ids: list[str] = []
    rack_cols: list[str] = []
    for idx, _zone in enumerate(zones):
        rack_id = f"synthetic_rack_{idx:03d}"
        col = f"{rack_id}_kw"
        out[col] = load
        rack_ids.append(rack_id)
        rack_cols.append(col)
    return rack_ids, rack_cols, out


def _rack_zone_map(config: dict[str, Any], rack_ids: list[str], zone_ids: list[str]) -> list[str]:
    fallback = zone_ids[0] if zone_ids else "zone_1"
    try:
        path = Path(config["paths"]["data_dir"]) / "rack_metadata.csv"
        metadata = pd.read_csv(path)
    except Exception:
        return [fallback for _ in rack_ids]
    if "rack_id" not in metadata.columns:
        return [fallback for _ in rack_ids]

    zone_set = set(zone_ids)
    by_rack = metadata.set_index("rack_id")
    mapped: list[str] = []
    for rack_id in rack_ids:
        zone = fallback
        lookup_id = rack_id
        room_id: str | None = None
        if rack_id not in by_rack.index and "_rack_it_" in rack_id:
            room_id, lookup_id = rack_id.split("_", 1)
        if lookup_id in by_rack.index:
            row = by_rack.loc[lookup_id]
            for col in ("ac_unit", "zone"):
                if col not in by_rack.columns:
                    continue
                candidate = str(row[col])
                room_candidate = f"{room_id}_{candidate}" if room_id else candidate
                if room_candidate in zone_set:
                    zone = room_candidate
                    break
                if candidate in zone_set:
                    zone = candidate
                    break
        mapped.append(zone)
    return mapped


def _heat_matrix(config: dict[str, Any], rack_count: int) -> np.ndarray:
    try:
        path = Path(config["paths"]["data_dir"]) / "H_matrix.csv"
        matrix = pd.read_csv(path, header=None).to_numpy(dtype=float)
    except Exception:
        return np.zeros((rack_count, rack_count), dtype=float)
    factor = int(round(float(config.get("scenario", {}).get("room_replication_factor", 1))))
    if factor > 1 and matrix.shape[0] == matrix.shape[1] and matrix.shape[0] * factor >= rack_count:
        matrix = np.kron(np.eye(factor), matrix)
    if matrix.shape[0] < rack_count or matrix.shape[1] < rack_count:
        return np.zeros((rack_count, rack_count), dtype=float)
    return np.nan_to_num(matrix[:rack_count, :rack_count], nan=0.0, posinf=0.0, neginf=0.0)


def _cyclic_groups(frame: pd.DataFrame) -> list[list[int]]:
    if "representative_day_id" not in frame.columns:
        return [list(range(len(frame)))]
    groups: list[list[int]] = []
    for _day_id, group in frame.groupby("representative_day_id", sort=False):
        groups.append(group.index.astype(int).tolist())
    return groups


def _supply_temp(
    decision: PlanningDecision,
    zone_id: str,
    old_supply_temp_c: float,
    new_supply_temp_c: float,
    aux_supply_temp_c: float,
) -> float:
    if decision.n_z.get(zone_id, 0):
        return new_supply_temp_c
    if decision.r_z.get(zone_id, 0):
        return aux_supply_temp_c
    if decision.o_z.get(zone_id, 0):
        return old_supply_temp_c
    return old_supply_temp_c
