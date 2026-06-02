from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from typical_days import TypicalDayData


def _gurobi_diagnostics() -> str:
    spec = importlib.util.find_spec('gurobipy')
    lines = [
        'Gurobi unavailable or unusable.',
        f'python_executable={sys.executable}',
        f'python_version={sys.version.split()[0]}',
        f'gurobipy_found={spec is not None}',
        f'gurobipy_location={spec.origin if spec else "N/A"}',
        f'GUROBI_HOME={os.environ.get("GUROBI_HOME", "")}',
        f'GRB_LICENSE_FILE={os.environ.get("GRB_LICENSE_FILE", "")}',
        f'PATH_contains_gurobi={"gurobi" in os.environ.get("PATH", "").lower()}',
    ]
    return '\n'.join(lines)


def require_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as e:
        raise RuntimeError(_gurobi_diagnostics() + f'\nimport_error={type(e).__name__}: {e}') from e
    return gp, GRB


@dataclass
class FixedPlanningDecision:
    y: np.ndarray
    x_ig: np.ndarray
    n_crac_remove: int
    cap_cdu_kw: float
    cap_hp_kw: float
    cap_bess_e_kwh: float
    cap_bess_p_kw: float
    cap_tes_kwh: float
    cap_tower_kw: float
    cap_hx_kw: float


@dataclass
class InnerSolveResult:
    feasible: bool
    status: str
    mip_gap: float
    solve_time_s: float
    objective_value: float
    operational_cost_yuan: float
    operational_carbon_kg: float
    heat_income_yuan: float
    hotspot_penalty_yuan: float
    grid_energy_kwh: float
    heat_supply_kwh: float
    task_balance_violation: float
    slot_violation: float
    thermal_violation: float
    power_balance_violation: float
    heat_balance_violation: float
    peak_day_violation: float
    diagnostic_slack_total: float
    series: Dict[str, np.ndarray]
    phase_times: Dict[str, float]
    build_mode: str = 'fresh'
    template_reuse_hit: bool = False
    template_build_once_s: float = 0.0
    template_update_s: float = 0.0


class InnerDispatchMILP:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model_table: Dict[str, np.ndarray],
        td: TypicalDayData,
        h_matrix: np.ndarray,
        fixed: FixedPlanningDecision,
        warm_start: Dict[str, np.ndarray] | None = None,
    ):
        self.cfg = cfg
        self.table = model_table
        self.td = td
        self.H = h_matrix
        self.fixed = fixed
        self.warm_start = warm_start
        self.use_hji = bool(cfg.get('model2', {}).get('thermal_topology', {}).get('use_hji', False))

        engine = str(cfg.get('solver', {}).get('inner', {}).get('engine', 'gurobi')).lower()
        if engine != 'gurobi':
            raise RuntimeError(
                f"Strict inner solver requires Gurobi only. Current solver.inner.engine='{engine}'"
            )

        self.gp, self.GRB = require_gurobi()
        try:
            self.model = self.gp.Model('inner_dispatch')
        except Exception as e:
            raise RuntimeError(_gurobi_diagnostics() + f"\nmodel_init_error={type(e).__name__}: {e}") from e

        self.sets: Dict[str, List[int]] = {}
        self.params: Dict[str, Any] = {}
        self.vars: Dict[str, Any] = {}
        self.constrs: Dict[str, Any] = {}
        diag_cfg = cfg.get('solver', {}).get('diagnostic_slack', {})
        self.diag_slack_enabled = bool(diag_cfg.get('enabled', True))
        self.diag_slack_eps = float(diag_cfg.get('eps', 1e-6))
        self.diag_slack_penalty = float(diag_cfg.get('penalty_yuan_per_unit', 1e9))
        if self.diag_slack_eps <= 0.0:
            self.diag_slack_enabled = False

    @staticmethod
    def _table_int(table: Dict[str, np.ndarray], key: str) -> int:
        v = table[key]
        if isinstance(v, np.ndarray):
            return int(np.asarray(v).reshape(-1)[0])
        return int(v)

    def _build_sets(self) -> None:
        n_r = self._table_int(self.table, 'n_racks')
        n_g = self._table_int(self.table, 'n_group')
        n_z = self._table_int(self.table, 'n_zone')
        n_k = self.td.n_scenarios
        n_t = self.td.horizon

        self.sets['R'] = list(range(n_r))
        self.sets['G'] = list(range(n_g))
        self.sets['Z'] = list(range(n_z))
        self.sets['K'] = list(range(n_k))
        self.sets['T'] = list(range(n_t))

    def _build_tsup(self, m2: Dict[str, Any]) -> np.ndarray:
        n_z = self._table_int(self.table, 'n_zone')
        n_k = self.td.n_scenarios
        n_t = self.td.horizon
        t_sup = np.resize(np.array(m2.get('t_sup_zone_c', [22.0] * n_z), dtype=float), n_z)
        return np.broadcast_to(t_sup.reshape(n_z, 1, 1), (n_z, n_k, n_t)).astype(float, copy=True)

    def _refresh_dynamic_params(self) -> None:
        ex = self.cfg['existing']
        qlc_coef_i = np.sum(self.params['eta_ig'] * self.fixed.x_ig * self.params['lambda_ig'], axis=1).astype(float)
        self.params['branch_cap_y'] = (self.params['branch_cap_i'] * self.fixed.y).astype(float)
        self.params['qlc_coef_i'] = qlc_coef_i
        self.params['crac_old_cap'] = float(
            (int(ex['n_crac_old']) - int(self.fixed.n_crac_remove)) * float(ex['cap_crac_unit_kw'])
        )
        self.params['cap_tower_total'] = float(self.params['tower_old_cap']) + float(self.fixed.cap_tower_kw)

    def _prepare_pwl_cache(self) -> None:
        m2 = self.cfg['model2']
        tower_curve = m2.get('tower_curve', {})
        cdu_curve = m2.get('cdu_curve', {})
        pwl_cfg = m2.get('pwl', {})

        seg_cdu = int(pwl_cfg.get('segments_cdu', 6))
        seg_tower = int(pwl_cfg.get('segments_tower', 6))
        xpts_cdu = np.linspace(0.0, 1.0, seg_cdu + 1)
        xpts_tower = np.linspace(0.0, 1.0, seg_tower + 1)

        p_cdu_idle = float(cdu_curve.get('p_idle_kw', 70.0))
        p_cdu_rated = float(cdu_curve.get('p_rated_kw', 850.0))
        a1 = float(cdu_curve.get('a1', 0.25))
        a2 = float(cdu_curve.get('a2', 0.35))
        a3 = float(cdu_curve.get('a3', 0.4))
        ypts_cdu = p_cdu_idle + p_cdu_rated * (a1 * xpts_cdu + a2 * xpts_cdu * xpts_cdu + a3 * xpts_cdu * xpts_cdu * xpts_cdu)

        p_tower_idle = float(tower_curve.get('p_idle_kw', 25.0))
        p_tower_fan_rated = float(tower_curve.get('p_fan_rated_kw', 620.0))
        alpha_wb = float(tower_curve.get('alpha_wb', 0.015))
        t_wb_ref = float(tower_curve.get('t_wb_ref_c', 20.0))
        eta_wb = 1.0 - alpha_wb * (self.params['t_wb'] - t_wb_ref)
        eta_wb = np.maximum(0.55, eta_wb)

        tower_ypts: List[List[List[float]]] = []
        for k in range(self.td.n_scenarios):
            row: List[List[float]] = []
            for t in range(self.td.horizon):
                row.append((p_tower_idle + p_tower_fan_rated * ((xpts_tower / float(eta_wb[k, t])) ** 3)).tolist())
            tower_ypts.append(row)

        self.params['pwl_xpts_cdu'] = xpts_cdu.tolist()
        self.params['pwl_ypts_cdu'] = ypts_cdu.tolist()
        self.params['pwl_xpts_tower'] = xpts_tower.tolist()
        self.params['pwl_ypts_tower'] = tower_ypts

    def _build_params(self) -> None:
        econ = self.cfg['economics']
        tech = self.cfg['technical']
        ex = self.cfg['existing']
        m2 = self.cfg['model2']

        rack_it = np.stack([self.td.series[c] for c in self.td.rack_load_cols], axis=0).astype(float)  # (R,K,T)

        cap_air_base = np.resize(
            np.array(m2.get('cap_air_zone_kw', [5000.0] * self._table_int(self.table, 'n_zone')), dtype=float),
            self._table_int(self.table, 'n_zone'),
        )
        cap_air_amp = float(m2.get('cap_air_daily_amp', 0.1))
        hours = np.arange(self.td.horizon, dtype=float)
        cap_mod = 1.0 + cap_air_amp * np.sin(2.0 * np.pi * ((hours % 24.0) - 7.0) / 24.0)
        cap_air = np.broadcast_to(
            cap_air_base.reshape(-1, 1, 1) * cap_mod.reshape(1, 1, -1),
            (self._table_int(self.table, 'n_zone'), self.td.n_scenarios, self.td.horizon),
        ).astype(float, copy=True)

        zone_of_rack = self.table['zone_of_rack'].astype(int)
        n_zone = self._table_int(self.table, 'n_zone')
        zone_racks = [np.where(zone_of_rack == z)[0].astype(int).tolist() for z in range(n_zone)]
        lambda_ig = self.table['lambda_ig'].astype(float)
        eta_ig = self.table['eta_ig'].astype(float)
        nonzero_h_cols: List[List[tuple[int, float]]] = []
        if self.use_hji:
            # H is sparse in practice, so store nonzero incoming neighbors once and reuse during constraint build.
            for i in range(self.H.shape[1]):
                nz = np.nonzero(self.H[:, i])[0]
                nonzero_h_cols.append([(int(j), float(self.H[j, i])) for j in nz])
        else:
            nonzero_h_cols = [[] for _ in range(self.H.shape[1])]

        self.params.update(
            {
                'weights_days': self.td.weights_days.astype(float),
                'is_peak': self.td.is_peak.astype(bool),
                'price_grid': self.td.series['price_grid'].astype(float),
                'cf_grid': self.td.series['cf_grid'].astype(float),
                'heat_demand': self.td.series['heat_demand_kw'].astype(float),
                'rack_it_ikt': rack_it,
                'a_ig': self.table['a_ig'].astype(float),
                'lambda_ig': lambda_ig,
                'eta_ig': eta_ig,
                'zone_of_rack': zone_of_rack,
                'zone_racks': zone_racks,
                'beta_i': self.table['beta_i'].astype(float),
                'tmax_i': self.table['tmax_i'].astype(float),
                'branch_cap_i': self.table['branch_cap_i'].astype(float),
                'tsup_zkt': self._build_tsup(m2),
                'cap_air_zkt': cap_air,
                'h_col_nz': nonzero_h_cols,
                't_wb': self.td.series['t_wb'].astype(float),
                'price_heat': float(econ['price_heat_yuan_per_kwh']),
                'vom': econ['vom'],
                'cop': tech['cop'],
                'eta': tech['eta'],
                'bess': tech['bess'],
                'tes': tech['tes'],
                'redundancy': tech['redundancy'],
                'chiller_old_cap': float(ex['cap_chiller_old_kw']),
                'tower_old_cap': float(ex['cap_tower_old_kw']),
                'lambda_hot': float(m2.get('lambda_hot', 1.0e4)),
            }
        )
        self._refresh_dynamic_params()
        self._prepare_pwl_cache()

    def _set_gurobi_params(self) -> None:
        inner = self.cfg.get('solver', {}).get('inner', {})
        self.model.Params.OutputFlag = int(inner.get('output_flag', 0))
        self.model.Params.TimeLimit = float(inner.get('time_limit', 180.0))
        self.model.Params.MIPGap = float(inner.get('mip_gap', 0.02))
        self.model.Params.Threads = int(inner.get('threads', 0))
        self.model.Params.NumericFocus = int(inner.get('numeric_focus', 1))
        if 'mip_focus' in inner:
            self.model.Params.MIPFocus = int(inner.get('mip_focus', 0))
        if 'nodefile_start_gb' in inner:
            self.model.Params.NodefileStart = float(inner.get('nodefile_start_gb', 0.0))
        if 'heuristics' in inner:
            self.model.Params.Heuristics = float(inner.get('heuristics', 0.05))
        if 'presolve' in inner:
            self.model.Params.Presolve = int(inner.get('presolve', -1))
        if 'cuts' in inner:
            self.model.Params.Cuts = int(inner.get('cuts', -1))
        if 'work_limit' in inner:
            self.model.Params.WorkLimit = float(inner.get('work_limit', 0.0))
        if 'dual_reductions' in inner:
            self.model.Params.DualReductions = int(inner.get('dual_reductions', 1))

    def _apply_warm_start(self) -> None:
        K, T = self.sets['K'], self.sets['T']
        n_k = len(K)
        n_t = len(T)
        warm_keys = [
            'q_crac',
            'q_low_hp',
            'q_hp_out',
            'q_tes_ch',
            'q_tes_dis',
            'e_tes',
            'p_bch',
            'p_bdis',
            'e_bess',
            'd_tes_ch',
            'd_tes_dis',
            'd_bch',
            'd_bdis',
            'p_grid',
        ]
        for key in warm_keys:
            if key not in self.vars:
                continue
            container = self.vars[key]
            for k in K:
                for t in T:
                    try:
                        container[k, t].Start = self.GRB.UNDEFINED
                    except Exception:
                        continue
        if not self.warm_start:
            return
        for key, values in self.warm_start.items():
            if key not in self.vars:
                continue
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size != n_k * n_t:
                continue
            arr2 = arr.reshape(n_k, n_t)
            container = self.vars[key]
            for k in K:
                for t in T:
                    try:
                        container[k, t].Start = float(arr2[k, t])
                    except Exception:
                        continue

    def _build_variables(self) -> None:
        R, G, Z, K, T = (self.sets[s] for s in ['R', 'G', 'Z', 'K', 'T'])
        m = self.model
        GRB = self.GRB

        self.vars['q_it'] = m.addVars(R, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_it')
        self.vars['q_lc'] = m.addVars(R, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_lc')
        self.vars['q_air'] = m.addVars(R, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_air')
        self.vars['s_air'] = m.addVars(Z, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='s_air')
        self.vars['theta'] = m.addVars(R, K, T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='theta')
        self.vars['xi'] = m.addVars(R, K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='xi')

        self.vars['p_grid'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_grid')
        self.vars['p_bch'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_bch')
        self.vars['p_bdis'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_bdis')
        self.vars['p_hp'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_hp')
        self.vars['p_crac'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_crac')
        self.vars['p_chiller'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_chiller')
        self.vars['p_cdu'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_cdu')
        self.vars['p_tower'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='p_tower')

        self.vars['q_hp_out'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_hp_out')
        self.vars['q_tes_ch'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_tes_ch')
        self.vars['q_tes_dis'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_tes_dis')
        self.vars['q_mid_hx'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_mid_hx')
        self.vars['q_mid_waste'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_mid_waste')
        self.vars['q_low_hp'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_low_hp')
        self.vars['q_crac'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_crac')
        self.vars['q_chiller'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_chiller')
        self.vars['q_chiller_tower'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_chiller_tower')
        self.vars['q_tower_total'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_tower_total')
        self.vars['q_source_mid'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_source_mid')
        self.vars['q_source_low'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_source_low')
        self.vars['q_heat_supply'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_heat_supply')
        self.vars['q_direct'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='q_direct')

        self.vars['e_tes'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='e_tes')
        self.vars['e_bess'] = m.addVars(K, T, lb=0.0, vtype=GRB.CONTINUOUS, name='e_bess')
        self.vars['d_tes_ch'] = m.addVars(K, T, vtype=GRB.BINARY, name='d_tes_ch')
        self.vars['d_tes_dis'] = m.addVars(K, T, vtype=GRB.BINARY, name='d_tes_dis')
        self.vars['d_bch'] = m.addVars(K, T, vtype=GRB.BINARY, name='d_bch')
        self.vars['d_bdis'] = m.addVars(K, T, vtype=GRB.BINARY, name='d_bdis')

        self.vars['phi_cdu'] = m.addVars(K, T, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='phi_cdu')
        self.vars['phi_tower'] = m.addVars(K, T, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='phi_tower')
        if self.diag_slack_enabled:
            ub = float(self.diag_slack_eps)
            self.vars['sl_cdu'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_cdu')
            self.vars['sl_crac'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_crac')
            self.vars['sl_chiller'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_chiller')
            self.vars['sl_tower'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_tower')
            self.vars['sl_power_pos'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_power_pos')
            self.vars['sl_power_neg'] = m.addVars(K, T, lb=0.0, ub=ub, vtype=GRB.CONTINUOUS, name='sl_power_neg')

    def _build_constraints(self) -> None:
        gp = self.gp
        R, Z, K, T = (self.sets[s] for s in ['R', 'Z', 'K', 'T'])
        p = self.params
        v = self.vars

        self.constrs['qit_fixed'] = self.model.addConstrs(
            (
                v['q_it'][i, k, t] == float(p['rack_it_ikt'][i, k, t])
                for i in R
                for k in K
                for t in T
            ),
            name='qit_fixed',
        )
        self.constrs['qlc'] = self.model.addConstrs(
            (
                v['q_lc'][i, k, t] == float(p['qlc_coef_i'][i]) * v['q_it'][i, k, t]
                for i in R
                for k in K
                for t in T
            ),
            name='qlc',
        )
        self.constrs['qair'] = self.model.addConstrs(
            (
                v['q_air'][i, k, t] == v['q_it'][i, k, t] - v['q_lc'][i, k, t]
                for i in R
                for k in K
                for t in T
            ),
            name='qair',
        )
        self.constrs['branch'] = self.model.addConstrs(
            (
                v['q_lc'][i, k, t] <= float(p['branch_cap_y'][i])
                for i in R
                for k in K
                for t in T
            ),
            name='branch',
        )

        if self.diag_slack_enabled:
            self.constrs['cdu_total'] = self.model.addConstrs(
                (
                    v['q_lc'].sum('*', k, t) <= float(self.fixed.cap_cdu_kw) + v['sl_cdu'][k, t]
                    for k in K
                    for t in T
                ),
                name='cdu_total',
            )
        else:
            self.constrs['cdu_total'] = self.model.addConstrs(
                (
                    v['q_lc'].sum('*', k, t) <= float(self.fixed.cap_cdu_kw)
                    for k in K
                    for t in T
                ),
                name='cdu_total',
            )

        self.constrs['air_load'] = self.model.addConstrs(
            (
                gp.quicksum(v['q_air'][i, k, t] for i in p['zone_racks'][z]) <= v['s_air'][z, k, t]
                for z in Z
                for k in K
                for t in T
            ),
            name='air_load',
        )
        self.constrs['air_cap'] = self.model.addConstrs(
            (
                v['s_air'][z, k, t] <= float(p['cap_air_zkt'][z, k, t])
                for z in Z
                for k in K
                for t in T
            ),
            name='air_cap',
        )

        self.constrs['qmid'] = self.model.addConstrs(
            (
                v['q_source_mid'][k, t] == v['q_lc'].sum('*', k, t)
                for k in K
                for t in T
            ),
            name='qmid',
        )
        self.constrs['qlow'] = self.model.addConstrs(
            (
                v['q_source_low'][k, t] == v['q_air'].sum('*', k, t)
                for k in K
                for t in T
            ),
            name='qlow',
        )

        if self.use_hji:
            self.constrs['theta'] = self.model.addConstrs(
                (
                    v['theta'][i, k, t]
                    == float(p['tsup_zkt'][int(p['zone_of_rack'][i]), k, t])
                    + gp.quicksum(coeff * v['q_air'][j, k, t] for j, coeff in p['h_col_nz'][i])
                    - float(p['beta_i'][i]) * v['s_air'][int(p['zone_of_rack'][i]), k, t]
                    for i in R
                    for k in K
                    for t in T
                ),
                name='theta',
            )
        else:
            self.constrs['theta'] = self.model.addConstrs(
                (
                    v['theta'][i, k, t]
                    == float(p['tsup_zkt'][int(p['zone_of_rack'][i]), k, t])
                    - float(p['beta_i'][i]) * v['s_air'][int(p['zone_of_rack'][i]), k, t]
                    for i in R
                    for k in K
                    for t in T
                ),
                name='theta',
            )
        self.constrs['thermal'] = self.model.addConstrs(
            (
                v['theta'][i, k, t] <= float(p['tmax_i'][i]) + v['xi'][i, k, t]
                for i in R
                for k in K
                for t in T
            ),
            name='thermal',
        )

        eta_hx = float(p['eta']['hx'])
        cop_hp = float(p['cop']['hp'])
        cop_crac = float(p['cop']['crac'])
        cop_chiller = float(p['cop']['chiller'])
        red_crac = float(p['redundancy']['crac'])
        red_chiller = float(p['redundancy'].get('chiller', 1.0))
        red_tower = float(p['redundancy']['tower'])
        cap_tower_total = float(p['cap_tower_total'])

        self.constrs['mid_split'] = self.model.addConstrs(
            (
                v['q_source_mid'][k, t] == v['q_mid_hx'][k, t] + v['q_mid_waste'][k, t]
                for k in K
                for t in T
            ),
            name='mid_split',
        )
        self.constrs['hx_cap'] = self.model.addConstrs(
            (
                v['q_mid_hx'][k, t] <= float(self.fixed.cap_hx_kw)
                for k in K
                for t in T
            ),
            name='hx_cap',
        )
        self.constrs['qdirect'] = self.model.addConstrs(
            (
                v['q_direct'][k, t] == eta_hx * v['q_mid_hx'][k, t]
                for k in K
                for t in T
            ),
            name='qdirect',
        )
        self.constrs['low_split'] = self.model.addConstrs(
            (
                v['q_source_low'][k, t] == v['q_low_hp'][k, t] + v['q_crac'][k, t]
                for k in K
                for t in T
            ),
            name='low_split',
        )
        self.constrs['hp_lift'] = self.model.addConstrs(
            (
                v['q_hp_out'][k, t] == v['q_low_hp'][k, t] * cop_hp / max(cop_hp - 1.0, 1e-9)
                for k in K
                for t in T
            ),
            name='hp_lift',
        )
        self.constrs['hp_cap'] = self.model.addConstrs(
            (
                v['q_hp_out'][k, t] <= float(self.fixed.cap_hp_kw)
                for k in K
                for t in T
            ),
            name='hp_cap',
        )
        self.constrs['php'] = self.model.addConstrs(
            (
                v['p_hp'][k, t] == v['q_hp_out'][k, t] / max(cop_hp, 1e-9)
                for k in K
                for t in T
            ),
            name='php',
        )
        self.constrs['pcrac'] = self.model.addConstrs(
            (
                v['p_crac'][k, t] == v['q_crac'][k, t] / max(cop_crac, 1e-9)
                for k in K
                for t in T
            ),
            name='pcrac',
        )
        self.constrs['qchiller'] = self.model.addConstrs(
            (
                v['q_chiller'][k, t] == v['q_crac'][k, t] + v['p_crac'][k, t]
                for k in K
                for t in T
            ),
            name='qchiller',
        )
        self.constrs['pchiller'] = self.model.addConstrs(
            (
                v['p_chiller'][k, t] == v['q_chiller'][k, t] / max(cop_chiller, 1e-9)
                for k in K
                for t in T
            ),
            name='pchiller',
        )
        if self.diag_slack_enabled:
            self.constrs['crac_cap'] = self.model.addConstrs(
                (
                    v['q_crac'][k, t] <= float(p['crac_old_cap']) / max(red_crac, 1e-9) + v['sl_crac'][k, t]
                    for k in K
                    for t in T
                ),
                name='crac_cap',
            )
            self.constrs['chiller_cap'] = self.model.addConstrs(
                (
                    v['q_chiller'][k, t] <= float(p['chiller_old_cap']) / max(red_chiller, 1e-9) + v['sl_chiller'][k, t]
                    for k in K
                    for t in T
                ),
                name='chiller_cap',
            )
        else:
            self.constrs['crac_cap'] = self.model.addConstrs(
                (
                    v['q_crac'][k, t] <= float(p['crac_old_cap']) / max(red_crac, 1e-9)
                    for k in K
                    for t in T
                ),
                name='crac_cap',
            )
            self.constrs['chiller_cap'] = self.model.addConstrs(
                (
                    v['q_chiller'][k, t] <= float(p['chiller_old_cap']) / max(red_chiller, 1e-9)
                    for k in K
                    for t in T
                ),
                name='chiller_cap',
            )
        self.constrs['tower_chiller'] = self.model.addConstrs(
            (
                v['q_chiller_tower'][k, t] == v['q_chiller'][k, t] + v['p_chiller'][k, t]
                for k in K
                for t in T
            ),
            name='tower_chiller',
        )
        self.constrs['tower_total'] = self.model.addConstrs(
            (
                v['q_tower_total'][k, t] == v['q_chiller_tower'][k, t] + v['q_mid_waste'][k, t]
                for k in K
                for t in T
            ),
            name='tower_total',
        )
        if self.diag_slack_enabled:
            self.constrs['tower_cap'] = self.model.addConstrs(
                (
                    v['q_tower_total'][k, t] <= cap_tower_total / max(red_tower, 1e-9) + v['sl_tower'][k, t]
                    for k in K
                    for t in T
                ),
                name='tower_cap',
            )
        else:
            self.constrs['tower_cap'] = self.model.addConstrs(
                (
                    v['q_tower_total'][k, t] <= cap_tower_total / max(red_tower, 1e-9)
                    for k in K
                    for t in T
                ),
                name='tower_cap',
            )
        self.constrs['phi_cdu'] = self.model.addConstrs(
            (
                v['q_source_mid'][k, t] == float(self.fixed.cap_cdu_kw) * v['phi_cdu'][k, t]
                for k in K
                for t in T
            ),
            name='phi_cdu',
        )
        self.constrs['phi_tower'] = self.model.addConstrs(
            (
                v['q_tower_total'][k, t] == cap_tower_total * v['phi_tower'][k, t]
                for k in K
                for t in T
            ),
            name='phi_tower',
        )

        self._build_pwl_constraints()

        eta_tes = float(p['eta']['tes_charge'])
        tes_sd = float(p['tes']['self_discharge_per_h'])
        gamma_tes = float(p['tes']['max_c_rate'])
        eta_bat = float(p['eta']['bess_charge'])
        bat_sd = float(p['bess']['self_discharge_per_h'])
        self.constrs['tes_dyn'] = self.model.addConstrs(
            (
                v['e_tes'][k, t]
                == v['e_tes'][k, (t - 1) % len(T)] * (1.0 - tes_sd)
                + v['q_tes_ch'][k, t] * eta_tes
                - v['q_tes_dis'][k, t] / max(float(p['eta']['tes_discharge']), 1e-9)
                for k in K
                for t in T
            ),
            name='tes_dyn',
        )
        self.constrs['tes_soc_max'] = self.model.addConstrs(
            (
                v['e_tes'][k, t] <= float(self.fixed.cap_tes_kwh)
                for k in K
                for t in T
            ),
            name='tes_soc_max',
        )
        self.constrs['tes_ch_cap'] = self.model.addConstrs(
            (
                v['q_tes_ch'][k, t] <= gamma_tes * float(self.fixed.cap_tes_kwh) * v['d_tes_ch'][k, t]
                for k in K
                for t in T
            ),
            name='tes_ch_cap',
        )
        self.constrs['tes_dis_cap'] = self.model.addConstrs(
            (
                v['q_tes_dis'][k, t] <= gamma_tes * float(self.fixed.cap_tes_kwh) * v['d_tes_dis'][k, t]
                for k in K
                for t in T
            ),
            name='tes_dis_cap',
        )
        self.constrs['tes_mode'] = self.model.addConstrs(
            (
                v['d_tes_ch'][k, t] + v['d_tes_dis'][k, t] <= 1
                for k in K
                for t in T
            ),
            name='tes_mode',
        )

        self.constrs['bess_dyn'] = self.model.addConstrs(
            (
                v['e_bess'][k, t]
                == v['e_bess'][k, (t - 1) % len(T)] * (1.0 - bat_sd)
                + v['p_bch'][k, t] * eta_bat
                - v['p_bdis'][k, t] / max(float(p['eta']['bess_discharge']), 1e-9)
                for k in K
                for t in T
            ),
            name='bess_dyn',
        )
        self.constrs['bess_soc_min'] = self.model.addConstrs(
            (
                v['e_bess'][k, t] >= 0.1 * float(self.fixed.cap_bess_e_kwh)
                for k in K
                for t in T
            ),
            name='bess_soc_min',
        )
        self.constrs['bess_soc_max'] = self.model.addConstrs(
            (
                v['e_bess'][k, t] <= 0.9 * float(self.fixed.cap_bess_e_kwh)
                for k in K
                for t in T
            ),
            name='bess_soc_max',
        )
        self.constrs['bch_cap'] = self.model.addConstrs(
            (
                v['p_bch'][k, t] <= float(self.fixed.cap_bess_p_kw) * v['d_bch'][k, t]
                for k in K
                for t in T
            ),
            name='bch_cap',
        )
        self.constrs['bdis_cap'] = self.model.addConstrs(
            (
                v['p_bdis'][k, t] <= float(self.fixed.cap_bess_p_kw) * v['d_bdis'][k, t]
                for k in K
                for t in T
            ),
            name='bdis_cap',
        )
        self.constrs['bmode'] = self.model.addConstrs(
            (
                v['d_bch'][k, t] + v['d_bdis'][k, t] <= 1
                for k in K
                for t in T
            ),
            name='bmode',
        )

        if self.diag_slack_enabled:
            self.constrs['power'] = self.model.addConstrs(
                (
                    v['p_grid'][k, t] + v['p_bdis'][k, t]
                    == gp.quicksum(v['q_it'][i, k, t] for i in R)
                    + v['p_crac'][k, t]
                    + v['p_chiller'][k, t]
                    + v['p_cdu'][k, t]
                    + v['p_tower'][k, t]
                    + v['p_hp'][k, t]
                    + v['p_bch'][k, t]
                    + v['sl_power_pos'][k, t]
                    - v['sl_power_neg'][k, t]
                    for k in K
                    for t in T
                ),
                name='power',
            )
        else:
            self.constrs['power'] = self.model.addConstrs(
                (
                    v['p_grid'][k, t] + v['p_bdis'][k, t]
                    == gp.quicksum(v['q_it'][i, k, t] for i in R)
                    + v['p_crac'][k, t]
                    + v['p_chiller'][k, t]
                    + v['p_cdu'][k, t]
                    + v['p_tower'][k, t]
                    + v['p_hp'][k, t]
                    + v['p_bch'][k, t]
                    for k in K
                    for t in T
                ),
                name='power',
            )

        self.constrs['heat_balance'] = self.model.addConstrs(
            (
                v['q_direct'][k, t] + v['q_hp_out'][k, t] + v['q_tes_dis'][k, t] == v['q_heat_supply'][k, t] + v['q_tes_ch'][k, t]
                for k in K
                for t in T
            ),
            name='heat_balance',
        )
        self.constrs['heat_demand_lb'] = self.model.addConstrs(
            (
                v['q_heat_supply'][k, t] >= float(p['heat_demand'][k, t])
                for k in K
                for t in T
            ),
            name='heat_demand_lb',
        )

    def _build_pwl_constraints(self) -> None:
        v = self.vars
        K, T = self.sets['K'], self.sets['T']
        xpts_cdu = self.params['pwl_xpts_cdu']
        ypts_cdu = self.params['pwl_ypts_cdu']
        xpts_tower = self.params['pwl_xpts_tower']
        ypts_tower = self.params['pwl_ypts_tower']
        for k in K:
            for t in T:
                self.model.addGenConstrPWL(v['phi_cdu'][k, t], v['p_cdu'][k, t], xpts_cdu, ypts_cdu, name=f'pwl_cdu_{k}_{t}')
                self.model.addGenConstrPWL(v['phi_tower'][k, t], v['p_tower'][k, t], xpts_tower, ypts_tower[k][t], name=f'pwl_tower_{k}_{t}')

    def _build_objective(self) -> None:
        R, K, T = self.sets['R'], self.sets['K'], self.sets['T']
        p = self.params
        v = self.vars

        w_day = p['weights_days']
        price_heat = float(p['price_heat'])
        lambda_hot = float(p['lambda_hot'])
        vom = p['vom']

        coef_grid = {}
        coef_heat_income = {}
        coef_om_cdu = {}
        coef_om_crac = {}
        coef_om_chiller = {}
        coef_om_hp = {}
        coef_om_tes = {}
        coef_hotspot = {}
        for k in K:
            for t in T:
                w = float(w_day[k])
                kt = (k, t)
                coef_grid[kt] = w * float(p['price_grid'][k, t])
                coef_heat_income[kt] = w * price_heat
                coef_om_cdu[kt] = w * float(vom['lc_yuan_per_kwh'])
                coef_om_crac[kt] = w * float(vom['air_yuan_per_kwh'])
                coef_om_chiller[kt] = w * float(vom['chiller_yuan_per_kwh'])
                coef_om_hp[kt] = w * float(vom['hp_yuan_per_kwh'])
                coef_om_tes[kt] = w * float(vom['tes_yuan_per_kwh'])
                for i in R:
                    coef_hotspot[(i, k, t)] = w * lambda_hot

        grid_cost = v['p_grid'].prod(coef_grid)
        om_cost = (
            v['p_cdu'].prod(coef_om_cdu)
            + v['p_crac'].prod(coef_om_crac)
            + v['p_chiller'].prod(coef_om_chiller)
            + v['p_hp'].prod(coef_om_hp)
            + v['q_tes_ch'].prod(coef_om_tes)
            + v['q_tes_dis'].prod(coef_om_tes)
        )
        heat_income = v['q_heat_supply'].prod(coef_heat_income)
        hotspot_penalty = v['xi'].prod(coef_hotspot)

        bess = self.cfg['technical']['bess']
        emb = self.cfg['carbon']['embodied']
        capex = self.cfg['economics']['capex']
        dod = float(bess['dod_max'])
        n_cycle = float(bess['n_cycle'])
        eta_dis = float(self.cfg['technical']['eta']['bess_discharge'])
        e_life = max(1e-9, self.fixed.cap_bess_e_kwh * n_cycle * dod * eta_dis)
        c_deg = float(capex['bess_energy_yuan_per_kwh']) * self.fixed.cap_bess_e_kwh / e_life
        self._ei_deg = float(emb['bess_energy_kg_per_kwh']) * self.fixed.cap_bess_e_kwh / e_life

        coef_bess_deg = {}
        for k in K:
            for t in T:
                coef_bess_deg[(k, t)] = float(w_day[k]) * c_deg
        self._coef_bess_deg = coef_bess_deg
        bess_deg_cost = v['p_bdis'].prod(coef_bess_deg)

        self._expr_grid_cost = grid_cost
        self._expr_om_cost = om_cost
        self._expr_heat_income = heat_income
        self._expr_hotspot_penalty = hotspot_penalty
        self._expr_bess_deg_cost = bess_deg_cost
        diag_slack_pen = 0.0
        if self.diag_slack_enabled:
            coef_diag = {}
            for k in K:
                for t in T:
                    coef_diag[(k, t)] = float(w_day[k]) * float(self.diag_slack_penalty)
            diag_slack_pen = (
                v['sl_cdu'].prod(coef_diag)
                + v['sl_crac'].prod(coef_diag)
                + v['sl_chiller'].prod(coef_diag)
                + v['sl_tower'].prod(coef_diag)
                + v['sl_power_pos'].prod(coef_diag)
                + v['sl_power_neg'].prod(coef_diag)
            )
        self._expr_diag_slack_penalty = diag_slack_pen

        obj = grid_cost + om_cost + bess_deg_cost - heat_income + hotspot_penalty + diag_slack_pen
        self.model.setObjective(obj, self.GRB.MINIMIZE)

    def _update_dynamic_model(self) -> None:
        p = self.params
        v = self.vars
        c = self.constrs
        R, K, T = (self.sets[s] for s in ['R', 'K', 'T'])

        self._refresh_dynamic_params()
        red_crac = float(p['redundancy']['crac'])
        red_tower = float(p['redundancy']['tower'])
        gamma_tes = float(p['tes']['max_c_rate'])
        cap_tower_total = float(p['cap_tower_total'])
        cap_tes = float(self.fixed.cap_tes_kwh)
        cap_bess_e = float(self.fixed.cap_bess_e_kwh)
        cap_bess_p = float(self.fixed.cap_bess_p_kw)
        cap_cdu = float(self.fixed.cap_cdu_kw)
        cap_hp = float(self.fixed.cap_hp_kw)
        cap_hx = float(self.fixed.cap_hx_kw)
        crac_cap_eff = float(p['crac_old_cap']) / max(red_crac, 1e-9)
        tower_cap_eff = cap_tower_total / max(red_tower, 1e-9)

        for i in R:
            coef = -float(p['qlc_coef_i'][i])
            rhs_branch = float(p['branch_cap_y'][i])
            for k in K:
                for t in T:
                    self.model.chgCoeff(c['qlc'][i, k, t], v['q_it'][i, k, t], coef)
                    c['branch'][i, k, t].RHS = rhs_branch
        for k in K:
            for t in T:
                c['cdu_total'][k, t].RHS = cap_cdu
                c['hx_cap'][k, t].RHS = cap_hx
                c['hp_cap'][k, t].RHS = cap_hp
                c['crac_cap'][k, t].RHS = crac_cap_eff
                c['tower_cap'][k, t].RHS = tower_cap_eff
                c['tes_soc_max'][k, t].RHS = cap_tes
                c['bess_soc_min'][k, t].RHS = 0.1 * cap_bess_e
                c['bess_soc_max'][k, t].RHS = 0.9 * cap_bess_e
                self.model.chgCoeff(c['phi_cdu'][k, t], v['phi_cdu'][k, t], -cap_cdu)
                self.model.chgCoeff(c['phi_tower'][k, t], v['phi_tower'][k, t], -cap_tower_total)
                self.model.chgCoeff(c['tes_ch_cap'][k, t], v['d_tes_ch'][k, t], -gamma_tes * cap_tes)
                self.model.chgCoeff(c['tes_dis_cap'][k, t], v['d_tes_dis'][k, t], -gamma_tes * cap_tes)
                self.model.chgCoeff(c['bch_cap'][k, t], v['d_bch'][k, t], -cap_bess_p)
                self.model.chgCoeff(c['bdis_cap'][k, t], v['d_bdis'][k, t], -cap_bess_p)
                v['p_bdis'][k, t].Obj = float(self._coef_bess_deg[(k, t)])
        bess = self.cfg['technical']['bess']
        emb = self.cfg['carbon']['embodied']
        capex = self.cfg['economics']['capex']
        dod = float(bess['dod_max'])
        n_cycle = float(bess['n_cycle'])
        eta_dis = float(self.cfg['technical']['eta']['bess_discharge'])
        e_life = max(1e-9, self.fixed.cap_bess_e_kwh * n_cycle * dod * eta_dis)
        c_deg = float(capex['bess_energy_yuan_per_kwh']) * self.fixed.cap_bess_e_kwh / e_life
        self._ei_deg = float(emb['bess_energy_kg_per_kwh']) * self.fixed.cap_bess_e_kwh / e_life
        for k in K:
            for t in T:
                v['p_bdis'][k, t].Obj = float(self.params['weights_days'][k]) * c_deg
        self._coef_bess_deg = {(k, t): float(self.params['weights_days'][k]) * c_deg for k in K for t in T}
        self._expr_bess_deg_cost = v['p_bdis'].prod(self._coef_bess_deg)
        obj = (
            self._expr_grid_cost
            + self._expr_om_cost
            + self._expr_bess_deg_cost
            - self._expr_heat_income
            + self._expr_hotspot_penalty
            + self._expr_diag_slack_penalty
        )
        self.model.setObjective(obj, self.GRB.MINIMIZE)
        self.model.update()

    def build_model_only(self) -> Dict[str, float]:
        phase_times: Dict[str, float] = {}
        t = perf_counter()
        self._set_gurobi_params()
        phase_times['set_params_s'] = perf_counter() - t

        t = perf_counter()
        self._build_sets()
        phase_times['build_sets_s'] = perf_counter() - t

        t = perf_counter()
        self._build_params()
        phase_times['build_params_s'] = perf_counter() - t

        t = perf_counter()
        self._build_variables()
        phase_times['build_variables_s'] = perf_counter() - t

        t = perf_counter()
        self._apply_warm_start()
        phase_times['apply_warm_start_s'] = perf_counter() - t

        t = perf_counter()
        self._build_constraints()
        phase_times['build_constraints_s'] = perf_counter() - t

        t = perf_counter()
        self._build_objective()
        phase_times['build_objective_s'] = perf_counter() - t

        phase_times['build_total_s'] = (
            phase_times['build_sets_s']
            + phase_times['build_params_s']
            + phase_times['build_variables_s']
            + phase_times['apply_warm_start_s']
            + phase_times['build_constraints_s']
            + phase_times['build_objective_s']
        )
        return phase_times

    def _optimize_and_collect(
        self,
        phase_times: Dict[str, float],
        t_all: float,
        build_mode: str = 'fresh',
        template_reuse_hit: bool = False,
        template_build_once_s: float = 0.0,
        template_update_s: float = 0.0,
    ) -> InnerSolveResult:
        t = perf_counter()
        self.model.optimize()
        elapsed = perf_counter() - t
        phase_times['optimize_s'] = elapsed
        phase_times['total_s'] = perf_counter() - t_all

        st = int(self.model.Status)
        feasible = st in (self.GRB.OPTIMAL, self.GRB.TIME_LIMIT, self.GRB.SUBOPTIMAL)
        if not feasible:
            return InnerSolveResult(
                feasible=False,
                status=f'gurobi_status_{st}',
                mip_gap=float('inf'),
                solve_time_s=elapsed,
                objective_value=float('inf'),
                operational_cost_yuan=float('inf'),
                operational_carbon_kg=float('inf'),
                heat_income_yuan=0.0,
                hotspot_penalty_yuan=float('inf'),
                grid_energy_kwh=float('inf'),
                heat_supply_kwh=0.0,
                task_balance_violation=float('inf'),
                slot_violation=float('inf'),
                thermal_violation=float('inf'),
                power_balance_violation=float('inf'),
                heat_balance_violation=float('inf'),
                peak_day_violation=float('inf'),
                diagnostic_slack_total=float('inf'),
                series={},
                phase_times=phase_times,
                build_mode=build_mode,
                template_reuse_hit=template_reuse_hit,
                template_build_once_s=float(template_build_once_s),
                template_update_s=float(template_update_s),
            )

        mip_gap = float(getattr(self.model, 'MIPGap', 0.0))
        diag_cost = float(self._expr_diag_slack_penalty.getValue()) if hasattr(self._expr_diag_slack_penalty, 'getValue') else float(self._expr_diag_slack_penalty)
        op_cost = float(self._expr_grid_cost.getValue() + self._expr_om_cost.getValue() + self._expr_bess_deg_cost.getValue() - self._expr_heat_income.getValue() + self._expr_hotspot_penalty.getValue() + diag_cost)
        heat_income = float(self._expr_heat_income.getValue())
        hotspot_pen = float(self._expr_hotspot_penalty.getValue())

        w = self.params['weights_days']
        v = self.vars
        K, T = self.sets['K'], self.sets['T']
        op_carbon = 0.0
        grid_energy = 0.0
        heat_supply = 0.0
        bess_deg_carbon = 0.0
        for k in K:
            for t in T:
                ww = float(w[k])
                pg = float(v['p_grid'][k, t].X)
                pd = float(v['p_bdis'][k, t].X)
                hs = float(v['q_heat_supply'][k, t].X)
                op_carbon += ww * float(self.params['cf_grid'][k, t]) * pg
                grid_energy += ww * pg
                heat_supply += ww * hs
                bess_deg_carbon += ww * self._ei_deg * pd
        carbon_offset = heat_supply * float(self.cfg['carbon']['boiler_cf_kg_per_kwh_heat']) / max(float(self.cfg['technical']['eta']['boiler']), 1e-9)
        op_carbon = op_carbon + bess_deg_carbon - carbon_offset

        return InnerSolveResult(
            feasible=True,
            status=f'gurobi_status_{st}',
            mip_gap=mip_gap,
            solve_time_s=elapsed,
            objective_value=float(self.model.ObjVal),
            operational_cost_yuan=op_cost,
            operational_carbon_kg=op_carbon,
            heat_income_yuan=heat_income,
            hotspot_penalty_yuan=hotspot_pen,
            grid_energy_kwh=grid_energy,
            heat_supply_kwh=heat_supply,
            task_balance_violation=self._calc_task_balance_violation(),
            slot_violation=self._calc_slot_violation(),
            thermal_violation=self._calc_thermal_violation(),
            power_balance_violation=self._calc_power_balance_violation(),
            heat_balance_violation=self._calc_heat_balance_violation(),
            peak_day_violation=self._calc_peak_day_violation(),
            diagnostic_slack_total=self._calc_diagnostic_slack_total(),
            series=self._extract_series(),
            phase_times=phase_times,
            build_mode=build_mode,
            template_reuse_hit=template_reuse_hit,
            template_build_once_s=float(template_build_once_s),
            template_update_s=float(template_update_s),
        )

    def solve(self) -> InnerSolveResult:
        t_all = perf_counter()
        phase_times = self.build_model_only()
        return self._optimize_and_collect(phase_times, t_all, build_mode='fresh', template_reuse_hit=False)

    def _extract_series(self) -> Dict[str, np.ndarray]:
        K, T = self.sets['K'], self.sets['T']
        v = self.vars
        p = self.params
        R = self.sets['R']

        n_k = len(K)
        n_t = len(T)
        out = {
            'scenario': np.repeat(np.arange(n_k, dtype=int), n_t),
            'hour': np.tile(np.arange(n_t, dtype=int), n_k),
        }

        keys = [
            'p_grid',
            'p_bch',
            'p_bdis',
            'p_hp',
            'p_crac',
            'p_chiller',
            'p_cdu',
            'p_tower',
            'q_tes_ch',
            'q_tes_dis',
            'q_heat_supply',
            'q_direct',
            'q_hp_out',
            'q_low_hp',
            'q_crac',
            'q_chiller',
            'e_tes',
            'e_bess',
        ]
        for key in keys:
            arr = np.zeros((n_k, n_t), dtype=float)
            for k in K:
                for t in T:
                    arr[k, t] = float(v[key][k, t].X)
            out[key] = arr.reshape(-1)

        it_total = np.zeros((n_k, n_t), dtype=float)
        heat_demand = np.array(p['heat_demand'], dtype=float)
        for k in K:
            for t in T:
                it_total[k, t] = sum(float(v['q_it'][i, k, t].X) for i in R)

        out['it_total'] = it_total.reshape(-1)
        out['heat_demand'] = heat_demand.reshape(-1)

        # Legacy-compatible aliases used by visualization / analysis scripts.
        out['p_grid_kw'] = out['p_grid']
        out['p_bess_ch_kw'] = out['p_bch']
        out['p_bess_dis_kw'] = out['p_bdis']
        out['p_hp_kw'] = out['p_hp']
        out['p_crac_kw'] = out['p_crac']
        out['p_chiller_kw'] = out['p_chiller']
        out['p_cdu_kw'] = out['p_cdu']
        out['p_tower_kw'] = out['p_tower']
        out['q_tes_ch_kw'] = out['q_tes_ch']
        out['q_tes_dis_kw'] = out['q_tes_dis']
        out['q_heat_supply_kw'] = out['q_heat_supply']
        out['q_heat_direct_kw'] = out['q_direct']
        out['q_heat_hp_kw'] = out['q_hp_out']
        out['q_chiller_kw'] = out['q_chiller']
        out['it_total_kw'] = out['it_total']
        out['heat_demand_kw'] = out['heat_demand']
        for key in ['d_tes_ch', 'd_tes_dis', 'd_bch', 'd_bdis']:
            arr = np.zeros((n_k, n_t), dtype=float)
            for k in K:
                for t in T:
                    arr[k, t] = float(v[key][k, t].X)
            out[key] = arr.reshape(-1)
        if self.diag_slack_enabled:
            for key in ['sl_cdu', 'sl_crac', 'sl_chiller', 'sl_tower', 'sl_power_pos', 'sl_power_neg']:
                arr = np.zeros((n_k, n_t), dtype=float)
                for k in K:
                    for t in T:
                        arr[k, t] = float(v[key][k, t].X)
                out[key] = arr.reshape(-1)
        return out

    def _calc_task_balance_violation(self) -> float:
        # Legacy output field: business allocation has been removed.
        return 0.0

    def _calc_slot_violation(self) -> float:
        # Legacy output field: rack IT load is now fixed from input time series.
        return 0.0

    def _calc_thermal_violation(self) -> float:
        R, K, T = (self.sets[s] for s in ['R', 'K', 'T'])
        theta = self.vars['theta']
        xi = self.vars['xi']
        tmax = self.params['tmax_i']
        mx = 0.0
        for i in R:
            for k in K:
                for t in T:
                    lhs = float(theta[i, k, t].X)
                    rhs = float(tmax[i]) + float(xi[i, k, t].X)
                    mx = max(mx, max(0.0, lhs - rhs))
        return mx

    def _calc_power_balance_violation(self) -> float:
        R, K, T = (self.sets[s] for s in ['R', 'K', 'T'])
        v = self.vars
        mx = 0.0
        for k in K:
            for t in T:
                p_it = sum(float(v['q_it'][i, k, t].X) for i in R)
                p_cool = float(v['p_crac'][k, t].X) + float(v['p_chiller'][k, t].X) + float(v['p_cdu'][k, t].X) + float(v['p_tower'][k, t].X)
                lhs = float(v['p_grid'][k, t].X) + float(v['p_bdis'][k, t].X)
                rhs = p_it + p_cool + float(v['p_hp'][k, t].X) + float(v['p_bch'][k, t].X)
                mx = max(mx, abs(lhs - rhs))
        return mx

    def _calc_heat_balance_violation(self) -> float:
        K, T = self.sets['K'], self.sets['T']
        v = self.vars
        mx = 0.0
        for k in K:
            for t in T:
                lhs = float(v['q_direct'][k, t].X) + float(v['q_hp_out'][k, t].X) + float(v['q_tes_dis'][k, t].X)
                rhs = float(v['q_heat_supply'][k, t].X) + float(v['q_tes_ch'][k, t].X)
                mx = max(mx, abs(lhs - rhs))
        return mx

    def _calc_peak_day_violation(self) -> float:
        K, T = self.sets['K'], self.sets['T']
        is_peak = self.params['is_peak']
        v = self.vars
        red_tower = float(self.params['redundancy']['tower'])
        cap_tower_effective = (float(self.params['tower_old_cap']) + float(self.fixed.cap_tower_kw)) / max(red_tower, 1e-9)
        mx = 0.0
        for k in K:
            if not bool(is_peak[k]):
                continue
            for t in T:
                mx = max(mx, max(0.0, float(v['q_source_mid'][k, t].X) - float(self.fixed.cap_cdu_kw)))
                mx = max(mx, max(0.0, float(v['q_tower_total'][k, t].X) - cap_tower_effective))
        return mx

    def _calc_diagnostic_slack_total(self) -> float:
        if not self.diag_slack_enabled:
            return 0.0
        K, T = self.sets['K'], self.sets['T']
        v = self.vars
        total = 0.0
        for k in K:
            for t in T:
                total += float(v['sl_cdu'][k, t].X)
                total += float(v['sl_crac'][k, t].X)
                total += float(v['sl_chiller'][k, t].X)
                total += float(v['sl_tower'][k, t].X)
                total += float(v['sl_power_pos'][k, t].X)
                total += float(v['sl_power_neg'][k, t].X)
        return total

    def diagnostic_slack_summary(self) -> Dict[str, float]:
        if not self.diag_slack_enabled:
            return {
                'cdu_sum': 0.0,
                'cdu_max': 0.0,
                'crac_sum': 0.0,
                'crac_max': 0.0,
                'chiller_sum': 0.0,
                'chiller_max': 0.0,
                'tower_sum': 0.0,
                'tower_max': 0.0,
                'power_sum': 0.0,
                'power_max': 0.0,
                'heat_sum': 0.0,
                'heat_max': 0.0,
                'soc_sum': 0.0,
                'soc_max': 0.0,
            }
        K, T = self.sets['K'], self.sets['T']
        v = self.vars

        def _sum_max(key: str) -> tuple[float, float]:
            vals = [float(v[key][k, t].X) for k in K for t in T]
            return float(np.sum(vals)), float(np.max(vals)) if vals else 0.0

        cdu_sum, cdu_max = _sum_max('sl_cdu')
        crac_sum, crac_max = _sum_max('sl_crac')
        chiller_sum, chiller_max = _sum_max('sl_chiller')
        tower_sum, tower_max = _sum_max('sl_tower')
        power_vals = [float(v['sl_power_pos'][k, t].X) + float(v['sl_power_neg'][k, t].X) for k in K for t in T]
        power_sum = float(np.sum(power_vals))
        power_max = float(np.max(power_vals)) if power_vals else 0.0
        return {
            'cdu_sum': cdu_sum,
            'cdu_max': cdu_max,
            'crac_sum': crac_sum,
            'crac_max': crac_max,
            'chiller_sum': chiller_sum,
            'chiller_max': chiller_max,
            'tower_sum': tower_sum,
            'tower_max': tower_max,
            'power_sum': power_sum,
            'power_max': power_max,
            # Current diagnostic slack model does not soften heat balance or SOC.
            'heat_sum': 0.0,
            'heat_max': 0.0,
            'soc_sum': 0.0,
            'soc_max': 0.0,
        }


class PersistentInnerDispatchModel:
    def __init__(
        self,
        cfg: Dict[str, Any],
        model_table: Dict[str, np.ndarray],
        td: TypicalDayData,
        h_matrix: np.ndarray,
    ):
        self.cfg = cfg
        self.model_table = model_table
        self.td = td
        self.h_matrix = h_matrix
        self.inner: InnerDispatchMILP | None = None
        self.template_solve_count = 0

    def solve(
        self,
        fixed: FixedPlanningDecision,
        warm_start: Dict[str, np.ndarray] | None = None,
    ) -> InnerSolveResult:
        t_all = perf_counter()
        phase_times: Dict[str, float] = {}
        build_mode = 'template_reuse'
        template_reuse_hit = self.inner is not None
        if self.inner is None:
            build_mode = 'template_initial_build'
            t = perf_counter()
            self.inner = InnerDispatchMILP(
                cfg=self.cfg,
                model_table=self.model_table,
                td=self.td,
                h_matrix=self.h_matrix,
                fixed=fixed,
                warm_start=warm_start,
            )
            build_phase = self.inner.build_model_only()
            phase_times.update(build_phase)
            phase_times['template_build_once_s'] = perf_counter() - t
            template_reuse_hit = False
        else:
            self.inner.fixed = fixed
            self.inner.warm_start = warm_start
            t = perf_counter()
            self.inner._update_dynamic_model()
            phase_times['template_update_s'] = perf_counter() - t
            t = perf_counter()
            self.inner._apply_warm_start()
            phase_times['apply_warm_start_s'] = perf_counter() - t
            phase_times['build_total_s'] = phase_times['template_update_s'] + phase_times['apply_warm_start_s']

        result = self.inner._optimize_and_collect(
            phase_times,
            t_all,
            build_mode=build_mode,
            template_reuse_hit=template_reuse_hit,
            template_build_once_s=float(phase_times.get('template_build_once_s', 0.0)),
            template_update_s=float(phase_times.get('template_update_s', 0.0)),
        )
        self.template_solve_count += 1
        return result
