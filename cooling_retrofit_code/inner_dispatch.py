from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    ) -> InnerSolveResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for inner MILP dispatch.") from exc

        frame = time_frame.reset_index(drop=True).copy()
        if frame.empty:
            return InnerSolveResult(False, float("inf"), pd.DataFrame(), {"reason": "empty_time_frame"})

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

        racks_by_zone = {zone: [idx for idx, z in enumerate(rack_to_zone) if z == zone] for zone in zones}
        q_air_zone = {(zone, t): float(q_air[t, racks_by_zone[zone]].sum()) for zone in zones for t in times}
        q_cp_zone = {(zone, t): float(q_cp[t, racks_by_zone[zone]].sum()) for zone in zones for t in times}
        q_rdhx_zone = {(zone, t): float(q_rdhx[t, racks_by_zone[zone]].sum()) for zone in zones for t in times}
        old_air_capacity_eff = {
            (zone, t): _time_series_value(frame, f"old_ac_capacity_eff_{zone}_kw", t, peak_load_by_zone_kw.get(zone, 0.0))
            for zone in zones
            for t in times
        }
        old_air_coeff_eff = {
            (zone, t): _time_series_value(frame, f"old_ac_terminal_coeff_{zone}_kw_per_kw", t, ac_fan_coeff)
            for zone in zones
            for t in times
        }
        psi_amb = {
            (zone, t): _time_series_value(frame, f"psi_amb_{zone}", t, 1.0)
            for zone in zones
            for t in times
        }

        model = gp.Model("new_research_inner_dispatch")
        model.Params.OutputFlag = 0
        if solver.get("time_limit_seconds"):
            model.Params.TimeLimit = float(solver["time_limit_seconds"])
        if solver.get("mip_gap") is not None:
            model.Params.MIPGap = float(solver["mip_gap"])
        if solver.get("threads") is not None:
            model.Params.Threads = int(solver["threads"])

        zt = [(z, t) for z in zones for t in times]
        s_air = model.addVars(zt, lb=0.0, name="s_air")
        s_air_old = model.addVars(zt, lb=0.0, name="s_air_old")
        s_air_new = model.addVars(zt, lb=0.0, name="s_air_new")
        q_air_to_ashp = model.addVars(zt, lb=0.0, name="q_air_to_ashp")
        q_air_to_reject = model.addVars(zt, lb=0.0, name="q_air_to_reject")
        q_cp_to_wshp = model.addVars(zt, lb=0.0, name="q_cp_to_wshp")
        q_cp_to_free = model.addVars(zt, lb=0.0, name="q_cp_to_free")
        q_rdhx_to_wshp = model.addVars(zt, lb=0.0, name="q_rdhx_to_wshp")
        q_rdhx_to_chiller = model.addVars(zt, lb=0.0, name="q_rdhx_to_chiller")

        q_ashp_in = model.addVars(times, lb=0.0, name="q_ashp_in")
        q_ashp_out = model.addVars(times, lb=0.0, ub=max(0.0, decision.cap_ashp), name="q_ashp_out")
        p_ashp = model.addVars(times, lb=0.0, name="p_ashp")
        q_wshp_in = model.addVars(times, lb=0.0, name="q_wshp_in")
        q_wshp_out = model.addVars(times, lb=0.0, ub=max(0.0, decision.cap_wshp), name="q_wshp_out")
        p_wshp = model.addVars(times, lb=0.0, name="p_wshp")
        q_chiller_evap = model.addVars(times, lb=0.0, ub=max(0.0, chiller_old_kw + decision.delta_cap_chiller), name="q_chiller_evap")
        p_chiller = model.addVars(times, lb=0.0, name="p_chiller")
        q_tower = model.addVars(times, lb=0.0, ub=max(0.0, tower_old_kw + decision.delta_cap_tower), name="q_tower")
        p_tower = model.addVars(times, lb=0.0, name="p_tower")
        p_pump = model.addVars(times, lb=0.0, name="p_pump")
        p_cdu_free = model.addVars(times, lb=0.0, name="p_cdu_free")
        p_terminal = model.addVars(times, lb=0.0, name="p_terminal")
        p_grid = model.addVars(times, lb=0.0, name="p_grid")
        q_heat = model.addVars(times, lb=0.0, name="q_heat")

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

        theta = model.addVars(range(len(rack_ids)), times, lb=-GRB.INFINITY, name="theta")
        xi = model.addVars(range(len(rack_ids)), times, lb=0.0, name="xi")

        for z in zones:
            for t in times:
                old_air_capacity = (
                    max(0.0, float(old_air_capacity_eff[z, t]) * float(psi_amb[z, t]))
                    if decision.o_z.get(z, 0)
                    else 0.0
                )
                new_air_capacity = float(decision.cap_ac_new[z])
                air_capacity = old_air_capacity + new_air_capacity
                model.addConstr(q_air_to_ashp[z, t] + q_air_to_reject[z, t] == q_air_zone[z, t])
                model.addConstr(q_air_to_ashp[z, t] <= decision.a_z.get(z, 0) * q_air_zone[z, t])
                model.addConstr(q_air_to_reject[z, t] <= air_capacity)
                model.addConstr(s_air[z, t] == s_air_old[z, t] + s_air_new[z, t])
                model.addConstr(s_air_old[z, t] <= old_air_capacity)
                model.addConstr(s_air_new[z, t] <= new_air_capacity)
                model.addConstr(s_air[z, t] >= q_air_zone[z, t])
                model.addConstr(s_air[z, t] <= air_capacity)
                model.addConstr(q_cp_to_wshp[z, t] + q_cp_to_free[z, t] == q_cp_zone[z, t])
                model.addConstr(q_cp_zone[z, t] <= float(decision.cap_cdu[z]) + 1.0e-7)
                model.addConstr(q_rdhx_to_wshp[z, t] + q_rdhx_to_chiller[z, t] == q_rdhx_zone[z, t])
                model.addConstr(q_rdhx_zone[z, t] <= float(decision.cap_rdhx[z]) + 1.0e-7)

        ashp_source_to_output = cop_ashp / max(cop_ashp - 1.0, 1.0e-9)
        wshp_source_to_output = cop_wshp / max(cop_wshp - 1.0, 1.0e-9)
        for t in times:
            model.addConstr(q_ashp_in[t] == gp.quicksum(q_air_to_ashp[z, t] for z in zones))
            model.addConstr(q_ashp_out[t] == ashp_source_to_output * q_ashp_in[t])
            model.addConstr(p_ashp[t] == q_ashp_out[t] / cop_ashp)
            model.addConstr(q_wshp_in[t] == gp.quicksum(q_cp_to_wshp[z, t] + q_rdhx_to_wshp[z, t] for z in zones))
            model.addConstr(q_wshp_out[t] == wshp_source_to_output * q_wshp_in[t])
            model.addConstr(p_wshp[t] == q_wshp_out[t] / cop_wshp)
            model.addConstr(q_chiller_evap[t] == gp.quicksum(q_air_to_reject[z, t] + q_rdhx_to_chiller[z, t] for z in zones))
            model.addConstr(p_chiller[t] == q_chiller_evap[t] / cop_chiller)
            model.addConstr(q_tower[t] == q_chiller_evap[t] + p_chiller[t] + gp.quicksum(q_cp_to_free[z, t] for z in zones))
            model.addConstr(p_tower[t] == tower_coeff * q_tower[t])
            model.addConstr(p_pump[t] == cdu_pump_coeff * gp.quicksum(q_cp_zone[z, t] for z in zones) + wshp_pump_coeff * q_wshp_in[t])
            model.addConstr(p_cdu_free[t] == cdu_free_coeff * gp.quicksum(q_cp_to_free[z, t] for z in zones))
            model.addConstr(
                p_terminal[t]
                == gp.quicksum(
                    old_air_coeff_eff[z, t] * s_air_old[z, t] + ac_fan_coeff * s_air_new[z, t]
                    for z in zones
                )
                + rdhx_coeff * gp.quicksum(q_rdhx_zone[z, t] for z in zones)
            )
            model.addConstr(q_ashp_out[t] + q_wshp_out[t] + q_tes_dis[t] == q_heat[t] + q_tes_ch[t])
            model.addConstr(q_heat[t] >= max(0.0, float(heat_demand[t])))
            model.addConstr(
                p_grid[t] + p_bess_dis[t]
                == float(it_total[t])
                + p_chiller[t]
                + p_tower[t]
                + p_ashp[t]
                + p_wshp[t]
                + p_pump[t]
                + p_cdu_free[t]
                + p_terminal[t]
                + p_bess_ch[t]
            )
            model.addConstr(q_tes_ch[t] <= tes_power_ub * d_tes_ch[t])
            model.addConstr(q_tes_dis[t] <= tes_power_ub * d_tes_dis[t])
            model.addConstr(d_tes_ch[t] + d_tes_dis[t] <= 1)
            model.addConstr(p_bess_ch[t] <= bess_charge_limit_kw * d_bess_ch[t])
            model.addConstr(p_bess_dis[t] <= bess_charge_limit_kw * d_bess_dis[t])
            model.addConstr(d_bess_ch[t] + d_bess_dis[t] <= 1)

        for day_indices in _cyclic_groups(frame):
            for pos, t in enumerate(day_indices):
                prev = day_indices[pos - 1]
                model.addConstr(e_tes[t] == (1.0 - tes_loss) * e_tes[prev] + tes_eta_ch * q_tes_ch[t] - q_tes_dis[t] / tes_eta_dis)
                model.addConstr(e_bess[t] == (1.0 - bess_loss) * e_bess[prev] + bess_eta_ch * p_bess_ch[t] - p_bess_dis[t] / bess_eta_dis)

        for i, zone_id in enumerate(rack_to_zone):
            supply_temp = _supply_temp(decision, zone_id, old_supply_temp_c, new_supply_temp_c, aux_supply_temp_c)
            for t in times:
                heat_influence = float(np.dot(heat_matrix[:, i], q_air[t, :]))
                model.addConstr(theta[i, t] == supply_temp + heat_influence - beta_air * s_air[zone_id, t])
                model.addConstr(theta[i, t] <= rack_temp_max_c + xi[i, t])

        model.setObjective(
            gp.quicksum(
                float(day_weight[t])
                * (
                    float(price[t]) * p_grid[t]
                    - heat_credit * q_heat[t]
                    + bess_deg_cost * p_bess_dis[t]
                    + hotspot_penalty * gp.quicksum(xi[i, t] for i in range(len(rack_ids)))
                )
                for t in times
            ),
            GRB.MINIMIZE,
        )
        model.optimize()

        if model.SolCount <= 0:
            return InnerSolveResult(
                feasible=False,
                objective_value=float("inf"),
                dispatch=pd.DataFrame(),
                diagnostics={
                    "gurobi_status": int(model.Status),
                    "reason": "no_solution",
                    "gurobi_time_limit_seconds": solver.get("time_limit_seconds"),
                    "gurobi_mip_gap": solver.get("mip_gap"),
                    "gurobi_threads": solver.get("threads"),
                    "model_sections": _model_sections(),
                },
            )

        rows: list[dict[str, Any]] = []
        max_power_residual = 0.0
        max_heat_residual = 0.0
        for t in times:
            q_air_total = sum(q_air_zone[z, t] for z in zones)
            q_cp_total = sum(q_cp_zone[z, t] for z in zones)
            q_rdhx_total = sum(q_rdhx_zone[z, t] for z in zones)
            s_air_total = sum(s_air[z, t].X for z in zones)
            s_air_old_total = sum(s_air_old[z, t].X for z in zones)
            s_air_new_total = sum(s_air_new[z, t].X for z in zones)
            old_air_capacity_total = sum(
                old_air_capacity_eff[z, t] * psi_amb[z, t] if decision.o_z.get(z, 0) else 0.0
                for z in zones
            )
            p_cooling = (
                p_chiller[t].X
                + p_tower[t].X
                + p_ashp[t].X
                + p_wshp[t].X
                + p_pump[t].X
                + p_cdu_free[t].X
                + p_terminal[t].X
            )
            power_residual = p_grid[t].X + p_bess_dis[t].X - float(it_total[t]) - p_cooling - p_bess_ch[t].X
            heat_residual = q_ashp_out[t].X + q_wshp_out[t].X + q_tes_dis[t].X - q_heat[t].X - q_tes_ch[t].X
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
                    "q_air_to_reject_kw": sum(q_air_to_reject[z, t].X for z in zones),
                    "q_cp_to_wshp_kw": sum(q_cp_to_wshp[z, t].X for z in zones),
                    "q_cp_to_free_kw": sum(q_cp_to_free[z, t].X for z in zones),
                    "q_rdhx_to_wshp_kw": sum(q_rdhx_to_wshp[z, t].X for z in zones),
                    "q_rdhx_to_chiller_kw": sum(q_rdhx_to_chiller[z, t].X for z in zones),
                    "q_ashp_out_kw": q_ashp_out[t].X,
                    "p_ashp_kw": p_ashp[t].X,
                    "q_wshp_in_kw": q_wshp_in[t].X,
                    "q_wshp_out_kw": q_wshp_out[t].X,
                    "p_wshp_kw": p_wshp[t].X,
                    "q_chiller_evap_kw": q_chiller_evap[t].X,
                    "p_chiller_kw": p_chiller[t].X,
                    "q_tower_kw": q_tower[t].X,
                    "p_tower_kw": p_tower[t].X,
                    "p_pump_kw": p_pump[t].X,
                    "p_cdu_free_kw": p_cdu_free[t].X,
                    "p_terminal_kw": p_terminal[t].X,
                    "p_cooling_kw": p_cooling,
                    "p_grid_kw": p_grid[t].X,
                    "q_heat_kw": q_heat[t].X,
                    "q_tes_ch_kw": q_tes_ch[t].X,
                    "q_tes_dis_kw": q_tes_dis[t].X,
                    "e_tes_kwh": e_tes[t].X,
                    "p_bess_ch_kw": p_bess_ch[t].X,
                    "p_bess_dis_kw": p_bess_dis[t].X,
                    "e_bess_kwh": e_bess[t].X,
                    "max_theta_c": max(theta[i, t].X for i in range(len(rack_ids))),
                    "hotspot_slack_c": sum(xi[i, t].X for i in range(len(rack_ids))),
                    "hotspot_penalty_yuan": hotspot_penalty * float(day_weight[t]) * sum(xi[i, t].X for i in range(len(rack_ids))),
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
            "max_hotspot_slack_c": float(dispatch["hotspot_slack_c"].max()),
            "total_hotspot_slack_c_hour": float((dispatch["hotspot_slack_c"] * dispatch["day_weight"]).sum()),
            "max_tes_soc_kwh": float(dispatch["e_tes_kwh"].max()),
            "max_bess_soc_kwh": float(dispatch["e_bess_kwh"].max()),
            "gurobi_time_limit_seconds": solver.get("time_limit_seconds"),
            "gurobi_mip_gap": solver.get("mip_gap"),
            "gurobi_threads": solver.get("threads"),
            "model_sections": _model_sections(),
        }
        return InnerSolveResult(True, float(model.ObjVal), dispatch, diagnostics)


def _gurobi_solver_config(config: dict[str, Any]) -> dict[str, Any]:
    solver = config.get("solver", {})
    gurobi = solver.get("gurobi", {}) if isinstance(solver.get("gurobi", {}), dict) else {}
    return {
        "time_limit_seconds": gurobi.get("time_limit_seconds", solver.get("time_limit_seconds")),
        "mip_gap": gurobi.get("mip_gap", solver.get("mip_gap")),
        "threads": gurobi.get("threads", solver.get("threads")),
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
