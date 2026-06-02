from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from typical_days import TypicalDayData


@dataclass
class OracleOutcome:
    feasible: bool
    status: str
    violation_total: float
    violations: Dict[str, float]
    n_crac_remove_raw: int
    n_crac_remove_fixed: int
    n_crac_remove_max: int
    caps_raw: Dict[str, float]
    caps_fixed: Dict[str, float]
    projection_applied: bool
    topology_cache_hit: bool
    reason_codes: List[str]
    metrics: Dict[str, float]


class FeasibilityOracle:
    def __init__(
        self,
        cfg: Dict[str, Any],
        table: Dict[str, np.ndarray],
        ts,
        rack_load_cols: List[str],
        td: TypicalDayData,
        cache_ctx: Dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.table = table
        self.ts = ts
        self.rack_load_cols = rack_load_cols
        self.td = td
        self.cache_ctx = cache_ctx

        fcfg = cfg.get("solver", {}).get("feasibility_oracle", {})
        self.enabled = bool(fcfg.get("enabled", True))
        self.violation_eps = float(fcfg.get("violation_eps", 1e-6))
        self.skip_inner_on_infeasible = bool(fcfg.get("skip_inner_on_infeasible", True))
        self.projection_enabled = bool(fcfg.get("projection_enabled", True))
        self.project_cdu = bool(fcfg.get("project_cap_cdu", True))
        self.project_hp = bool(fcfg.get("project_cap_hp", True))
        self.project_tower = bool(fcfg.get("project_cap_tower", True))
        self.project_n_crac_remove = bool(fcfg.get("project_n_crac_remove", True))

        self.n_old_crac = int(cfg["existing"]["n_crac_old"])
        self.crac_unit = float(cfg["existing"]["cap_crac_unit_kw"])
        self.cap_chiller_old = float(cfg["existing"]["cap_chiller_old_kw"])
        self.cap_tower_old = float(cfg["existing"]["cap_tower_old_kw"])
        self.red_crac = float(cfg["technical"]["redundancy"]["crac"])
        self.red_tower = float(cfg["technical"]["redundancy"]["tower"])
        self.cop_hp = float(cfg["technical"]["cop"]["hp"])
        self.cop_crac = float(cfg["technical"]["cop"]["crac"])
        self.cop_chiller = float(cfg["technical"]["cop"]["chiller"])

        self.lambda_ig = np.asarray(table["lambda_ig"], dtype=float)
        self.eta_ig = np.asarray(table["eta_ig"], dtype=float)

        # Build period-level arrays on typical-day horizon.
        self.n_k = int(td.n_scenarios)
        self.n_t = int(td.horizon)
        self.n_period = self.n_k * self.n_t
        self.hour_idx = np.tile(np.arange(self.n_t, dtype=int), self.n_k)
        self.scenario_idx = np.repeat(np.arange(self.n_k, dtype=int), self.n_t)

        rack_loads = [np.asarray(td.series[c], dtype=float).reshape(-1) for c in rack_load_cols]
        self.rack_it_ir = np.stack(rack_loads, axis=0)  # (R, N)
        self.it_total = np.sum(self.rack_it_ir, axis=0)
        self.low_base = self.it_total

        db = cfg.get("decision_bounds", {})
        self.cap_bounds = {
            "cap_cdu_kw": tuple(db.get("cap_cdu_kw", [0.0, 1e9])),
            "cap_hp_kw": tuple(db.get("cap_hp_kw", [0.0, 1e9])),
            "cap_tower_kw": tuple(db.get("cap_tower_kw", [0.0, 1e9])),
        }

        self._topology_cache = self._get_topology_cache(cache_ctx)

    @staticmethod
    def _get_topology_cache(cache_ctx: Dict[str, Any] | None):
        if cache_ctx is None:
            return {}
        if "oracle_topology_cache" not in cache_ctx:
            cache_ctx["oracle_topology_cache"] = {}
        return cache_ctx["oracle_topology_cache"]

    @staticmethod
    def topology_key(y: np.ndarray, x_ig: np.ndarray) -> Tuple[int, ...]:
        arr = np.concatenate([y.reshape(-1), x_ig.reshape(-1)], axis=0)
        return tuple(np.round(arr).astype(int).tolist())

    def _clip_cap(self, name: str, val: float) -> float:
        lo, hi = self.cap_bounds.get(name, (0.0, 1e12))
        return float(np.clip(val, lo, hi))

    def _get_topology_envelope(self, y: np.ndarray, x_ig: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool]:
        key = self.topology_key(y, x_ig)
        if key in self._topology_cache:
            return self._topology_cache[key], True

        coef_i = np.sum(self.eta_ig * np.asarray(x_ig, dtype=float) * self.lambda_ig, axis=1)
        coef_i = np.clip(coef_i, 0.0, None)
        q_lc_ts = self.rack_it_ir.T @ coef_i
        env = {"q_lc_min_ts": q_lc_ts.astype(float), "q_lc_max_ts": q_lc_ts.astype(float)}
        env["coef_i"] = coef_i
        env["key"] = key
        self._topology_cache[key] = env
        return env, False

    def assess(
        self,
        y: np.ndarray,
        x_ig: np.ndarray,
        n_crac_remove_raw: int,
        caps_raw: Dict[str, float],
    ) -> OracleOutcome:
        env, hit = self._get_topology_envelope(y, x_ig)
        eps = self.violation_eps
        caps = dict(caps_raw)
        projection_applied = False

        q_lc_min_ts = np.asarray(env["q_lc_min_ts"], dtype=float)
        q_lc_max_ts = np.asarray(env["q_lc_max_ts"], dtype=float)

        # 1) CDU chain
        q_lc_min_max = float(np.max(q_lc_min_ts))
        if self.projection_enabled and self.project_cdu and caps["cap_cdu_kw"] + eps < q_lc_min_max:
            caps["cap_cdu_kw"] = self._clip_cap("cap_cdu_kw", q_lc_min_max)
            projection_applied = True
        v_cdu = max(0.0, q_lc_min_max - float(caps["cap_cdu_kw"]))

        # 2) low-temperature chain (HP + CRAC)
        q_lc_used_max = np.minimum(float(caps["cap_cdu_kw"]), q_lc_max_ts)
        q_low_hp_max = max(0.0, float(caps["cap_hp_kw"]) * max(self.cop_hp - 1.0, 0.0) / max(self.cop_hp, 1e-9))
        q_crac_lb = np.maximum(0.0, self.low_base - q_lc_used_max - q_low_hp_max)
        q_crac_lb_max = float(np.max(q_crac_lb))

        n_raw = int(np.clip(n_crac_remove_raw, 0, self.n_old_crac))
        n_remove_max = int(np.floor(self.n_old_crac - self.red_crac * q_crac_lb_max / max(self.crac_unit, 1e-9) + 1e-9))
        n_remove_max = int(np.clip(n_remove_max, 0, self.n_old_crac))
        if self.project_n_crac_remove:
            n_fixed = min(n_raw, n_remove_max)
        else:
            n_fixed = n_raw
        if n_fixed != n_raw:
            projection_applied = True

        cap_crac_eff = (self.n_old_crac - n_fixed) * self.crac_unit / max(self.red_crac, 1e-9)
        v_crac = max(0.0, q_crac_lb_max - cap_crac_eff)

        if self.projection_enabled and self.project_hp and v_crac > eps:
            req_q_low_hp = float(np.max(np.maximum(0.0, self.low_base - q_lc_used_max - cap_crac_eff)))
            hp_req = req_q_low_hp * self.cop_hp / max(self.cop_hp - 1.0, 1e-9)
            hp_new = self._clip_cap("cap_hp_kw", max(float(caps["cap_hp_kw"]), hp_req))
            if hp_new > float(caps["cap_hp_kw"]) + eps:
                caps["cap_hp_kw"] = hp_new
                projection_applied = True
                q_low_hp_max = max(0.0, float(caps["cap_hp_kw"]) * max(self.cop_hp - 1.0, 0.0) / max(self.cop_hp, 1e-9))
                q_crac_lb = np.maximum(0.0, self.low_base - q_lc_used_max - q_low_hp_max)
                q_crac_lb_max = float(np.max(q_crac_lb))
                n_remove_max = int(np.floor(self.n_old_crac - self.red_crac * q_crac_lb_max / max(self.crac_unit, 1e-9) + 1e-9))
                n_remove_max = int(np.clip(n_remove_max, 0, self.n_old_crac))
                n_fixed = min(n_raw, n_remove_max) if self.project_n_crac_remove else n_raw
                cap_crac_eff = (self.n_old_crac - n_fixed) * self.crac_unit / max(self.red_crac, 1e-9)
                v_crac = max(0.0, q_crac_lb_max - cap_crac_eff)

        # 3) Chiller/Tower chain
        q_chiller_lb = q_crac_lb * (1.0 + 1.0 / max(self.cop_crac, 1e-9))
        q_chiller_lb_max = float(np.max(q_chiller_lb))
        v_chiller = max(0.0, q_chiller_lb_max - self.cap_chiller_old)

        q_tower_lb = q_chiller_lb * (1.0 + 1.0 / max(self.cop_chiller, 1e-9))
        q_tower_lb_max = float(np.max(q_tower_lb))
        cap_tower_eff = (self.cap_tower_old + float(caps["cap_tower_kw"])) / max(self.red_tower, 1e-9)
        v_tower = max(0.0, q_tower_lb_max - cap_tower_eff)
        if self.projection_enabled and self.project_tower and v_tower > eps:
            delta_req = max(0.0, self.red_tower * q_tower_lb_max - self.cap_tower_old)
            tower_new = self._clip_cap("cap_tower_kw", max(float(caps["cap_tower_kw"]), delta_req))
            if tower_new > float(caps["cap_tower_kw"]) + eps:
                caps["cap_tower_kw"] = tower_new
                projection_applied = True
                cap_tower_eff = (self.cap_tower_old + float(caps["cap_tower_kw"])) / max(self.red_tower, 1e-9)
                v_tower = max(0.0, q_tower_lb_max - cap_tower_eff)

        violations = {
            "cdu": float(v_cdu),
            "crac": float(v_crac),
            "chiller": float(v_chiller),
            "tower": float(v_tower),
        }
        violation_total = float(sum(violations.values()))
        feasible = bool(violation_total <= eps and np.isfinite(violation_total))

        parts = [k for k, v in violations.items() if v > eps]
        status = "oracle_pass" if feasible else ("oracle_infeasible_" + "_".join(parts))
        reason_codes: List[str] = []
        if np.any(~np.isfinite(q_lc_min_ts)) or np.any(~np.isfinite(q_lc_max_ts)):
            reason_codes.append("topology_envelope_failed")
        if v_cdu > eps:
            reason_codes.append("cdu_capacity")
        if v_crac > eps:
            reason_codes.append("crac_capacity")
        if v_chiller > eps:
            reason_codes.append("chiller_capacity_fixed_infrastructure")
        if v_tower > eps:
            reason_codes.append("tower_capacity")
        for name, raw_val in caps_raw.items():
            fixed_val = float(caps.get(name, raw_val))
            _, hi = self.cap_bounds.get(name, (0.0, float("inf")))
            if fixed_val >= float(hi) - eps and fixed_val > float(raw_val) + eps:
                reason_codes.append("projection_bound_hit")
                break
        bound_checks = [
            ("cap_cdu_kw", v_cdu),
            ("cap_hp_kw", v_crac),
            ("cap_tower_kw", v_tower),
        ]
        if "projection_bound_hit" not in reason_codes:
            for name, violation in bound_checks:
                _, hi = self.cap_bounds.get(name, (0.0, float("inf")))
                if violation > eps and float(caps.get(name, 0.0)) >= float(hi) - eps:
                    reason_codes.append("projection_bound_hit")
                    break

        idx_lc_peak = int(np.argmax(q_lc_min_ts))
        idx_crac_peak = int(np.argmax(q_crac_lb))
        idx_tower_peak = int(np.argmax(q_tower_lb))
        metrics = {
            "q_lc_min_max_kw": q_lc_min_max,
            "q_lc_max_max_kw": float(np.max(q_lc_max_ts)),
            "q_crac_lb_max_kw": q_crac_lb_max,
            "q_chiller_lb_max_kw": q_chiller_lb_max,
            "q_tower_lb_max_kw": q_tower_lb_max,
            "cap_crac_eff_kw": float(cap_crac_eff),
            "cap_tower_eff_kw": float(cap_tower_eff),
            "peak_period_lc_idx": float(idx_lc_peak),
            "peak_period_crac_idx": float(idx_crac_peak),
            "peak_period_tower_idx": float(idx_tower_peak),
            "peak_scenario_lc": float(self.scenario_idx[idx_lc_peak]),
            "peak_hour_lc": float(self.hour_idx[idx_lc_peak]),
            "peak_scenario_crac": float(self.scenario_idx[idx_crac_peak]),
            "peak_hour_crac": float(self.hour_idx[idx_crac_peak]),
            "peak_scenario_tower": float(self.scenario_idx[idx_tower_peak]),
            "peak_hour_tower": float(self.hour_idx[idx_tower_peak]),
        }

        return OracleOutcome(
            feasible=feasible,
            status=status,
            violation_total=violation_total,
            violations=violations,
            n_crac_remove_raw=n_raw,
            n_crac_remove_fixed=n_fixed,
            n_crac_remove_max=n_remove_max,
            caps_raw=dict(caps_raw),
            caps_fixed=caps,
            projection_applied=projection_applied,
            topology_cache_hit=hit,
            reason_codes=reason_codes,
            metrics=metrics,
        )

