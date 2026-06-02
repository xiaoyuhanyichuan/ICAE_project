from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from feasibility_oracle import FeasibilityOracle
from inner_dispatch import FixedPlanningDecision, InnerDispatchMILP, PersistentInnerDispatchModel, require_gurobi
from typical_days import TypicalDayBuilder, TypicalDayData


ROOT = Path(__file__).resolve().parents[1]
_MODEL2_CACHE: Dict[Tuple[int, int, int, int], Dict[str, np.ndarray]] = {}
_PROFILE_COUNTERS: Dict[str, int] = {
    'typical_day_build_count': 0,
}


@dataclass
class DecisionSchema:
    names: List[str]
    bounds: List[Tuple[float, float]]
    int_idx: List[int]
    bin_idx: List[int]


def crf(rate: float, years: int) -> float:
    if years <= 0:
        return 1.0
    if abs(rate) < 1e-9:
        return 1.0 / years
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def reset_profile_counters() -> None:
    for k in list(_PROFILE_COUNTERS.keys()):
        _PROFILE_COUNTERS[k] = 0


def get_profile_counters() -> Dict[str, int]:
    return {k: int(v) for k, v in _PROFILE_COUNTERS.items()}


def _count_profile(name: str, inc: int = 1) -> None:
    _PROFILE_COUNTERS[name] = int(_PROFILE_COUNTERS.get(name, 0)) + int(inc)


def _get_model2_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get('model2', {})


def _get_total_racks(cfg: Dict[str, Any]) -> int:
    ex = cfg.get('existing', {})
    return int(ex['n_racks_total'])


def _get_model2_table(cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    ex = cfg['existing']
    m2 = _get_model2_params(cfg)
    n_racks = _get_total_racks(cfg)
    n_zone = int(m2.get('n_zone', 4))
    n_group = int(m2.get('n_group', 3))
    seed = int(m2.get('seed', cfg.get('project', {}).get('random_seed', 42) + 1000))
    key = (n_racks, n_zone, n_group, seed)
    if key in _MODEL2_CACHE:
        return _MODEL2_CACHE[key]

    rng = np.random.default_rng(seed)
    compat_base = np.resize(np.array(m2.get('compat_prob', [0.9, 0.65, 0.25]), dtype=float), n_group)
    eta_base = np.resize(np.array(m2.get('eta_lc_group', [0.9, 0.78, 0.62]), dtype=float), n_group)
    server_groups = m2.get('server_groups', {})
    lam_base = np.resize(
        np.array(server_groups.get('heat_share', [0.5, 0.3, 0.2]), dtype=float),
        n_group,
    )
    lam_base = np.maximum(lam_base, 1e-6)
    lam_base /= np.sum(lam_base)

    a_ig = (rng.random((n_racks, n_group)) < compat_base.reshape(1, -1)).astype(float)
    eta_ig = np.clip(rng.normal(eta_base.reshape(1, -1), 0.04, size=(n_racks, n_group)), 0.45, 0.98)
    heat_share_jitter = float(server_groups.get('heat_share_jitter', 0.08))
    lambda_ig = np.clip(lam_base.reshape(1, -1) * (1 + rng.normal(0.0, heat_share_jitter, size=(n_racks, n_group))), 0.02, None)
    lambda_ig /= lambda_ig.sum(axis=1, keepdims=True)

    zone_of_rack = rng.integers(0, n_zone, size=n_racks)
    hotspot_risk = np.clip(rng.normal(1.0, 0.2, size=n_racks), 0.6, 1.7)
    branch_factor = np.clip(rng.normal(m2.get('branch_cap_factor_mean', 1.25), 0.15, size=n_racks), 0.8, 1.8)
    branch_cap_i = branch_factor * np.clip(rng.normal(float(m2.get('branch_cap_kw_mean', 120.0)), 18.0, size=n_racks), 70.0, None)

    c_fix = np.clip(rng.normal(m2.get('c_fix_mean', 18000.0), m2.get('c_fix_std', 3500.0), size=n_racks), 8000.0, None)
    c_plate = np.clip(rng.normal(m2.get('c_plate_mean', 2600.0), m2.get('c_plate_std', 400.0), size=(n_racks, n_group)), 1200.0, None)
    c_pipe_u = np.clip(rng.normal(m2.get('c_pipe_unit_mean', 220.0), m2.get('c_pipe_unit_std', 35.0), size=n_racks), 100.0, None)
    l_pipe = np.clip(rng.normal(m2.get('pipe_len_mean', 24.0), m2.get('pipe_len_std', 8.0), size=n_racks), 5.0, None)
    c_down = np.clip(rng.normal(m2.get('c_downtime_mean', 7500.0), m2.get('c_downtime_std', 1600.0), size=n_racks), 1500.0, None)

    ei_fix = np.clip(rng.normal(m2.get('ei_fix_mean', 420.0), m2.get('ei_fix_std', 70.0), size=n_racks), 150.0, None)
    ei_plate = np.clip(rng.normal(m2.get('ei_plate_mean', 38.0), m2.get('ei_plate_std', 8.0), size=(n_racks, n_group)), 10.0, None)
    ei_pipe_u = np.clip(rng.normal(m2.get('ei_pipe_unit_mean', 9.0), m2.get('ei_pipe_unit_std', 2.0), size=n_racks), 2.0, None)
    ei_down = np.clip(rng.normal(m2.get('ei_downtime_mean', 45.0), m2.get('ei_downtime_std', 12.0), size=n_racks), 8.0, None)

    w_old = np.clip(rng.normal(m2.get('w_old_mean', 850.0), m2.get('w_old_std', 120.0), size=n_racks), 500.0, None)
    w_fix = np.clip(rng.normal(m2.get('w_fix_mean', 55.0), m2.get('w_fix_std', 12.0), size=n_racks), 20.0, None)
    w_plate = np.clip(rng.normal(m2.get('w_plate_mean', 7.5), m2.get('w_plate_std', 2.0), size=(n_racks, n_group)), 2.0, None)
    w_rack_max = np.clip(rng.normal(m2.get('w_rack_max_mean', 1300.0), m2.get('w_rack_max_std', 80.0), size=n_racks), 1050.0, None)
    w_point_max = np.clip(rng.normal(m2.get('w_point_max_mean', 1250.0), m2.get('w_point_max_std', 80.0), size=n_racks), 1000.0, None)

    beta_i = np.resize(np.array(m2.get('beta_zone', [0.0012] * n_zone), dtype=float), n_zone)[zone_of_rack]
    tmax_i = np.full(n_racks, float(m2.get('t_inlet_max_c', 27.0)), dtype=float)

    table = {
        'n_racks': np.array([n_racks], dtype=int),
        'n_group': np.array([n_group], dtype=int),
        'n_zone': np.array([n_zone], dtype=int),
        'a_ig': a_ig,
        'eta_ig': eta_ig,
        'lambda_ig': lambda_ig,
        'zone_of_rack': zone_of_rack,
        'beta_i': beta_i,
        'tmax_i': tmax_i,
        'branch_cap_i': branch_cap_i,
        'hotspot_risk': hotspot_risk,
        'c_fix': c_fix,
        'c_plate': c_plate,
        'c_pipe_u': c_pipe_u,
        'l_pipe': l_pipe,
        'c_down': c_down,
        'ei_fix': ei_fix,
        'ei_plate': ei_plate,
        'ei_pipe_u': ei_pipe_u,
        'ei_down': ei_down,
        'w_old': w_old,
        'w_fix': w_fix,
        'w_plate': w_plate,
        'w_rack_max': w_rack_max,
        'w_point_max': w_point_max,
    }
    _MODEL2_CACHE[key] = table
    return table


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _load_h_matrix(cfg: Dict[str, Any], n_racks: int) -> np.ndarray:
    topo = cfg.get('model2', {}).get('thermal_topology', {})
    h_path_raw = topo.get('h_matrix_path', None)
    if not h_path_raw:
        raise RuntimeError('Missing required config: model2.thermal_topology.h_matrix_path')
    h_path = _resolve_path(str(h_path_raw))
    if not h_path.exists():
        raise RuntimeError(f'H matrix file not found: {h_path}')

    df = pd.read_csv(h_path, header=None)
    mat = df.to_numpy(dtype=float)
    if mat.shape == (n_racks + 1, n_racks + 1):
        mat = mat[1:, 1:]
    elif mat.shape == (n_racks, n_racks + 1):
        mat = mat[:, 1:]
    elif mat.shape == (n_racks + 1, n_racks):
        mat = mat[1:, :]

    if mat.shape != (n_racks, n_racks):
        raise RuntimeError(f'H matrix shape mismatch: expected {(n_racks, n_racks)}, got {mat.shape}')
    return mat


def _use_hji(cfg: Dict[str, Any]) -> bool:
    topo = cfg.get('model2', {}).get('thermal_topology', {})
    return bool(topo.get('use_hji', False))


def _get_rack_load_cols(cfg: Dict[str, Any], ts: pd.DataFrame, n_racks: int) -> List[str]:
    require_gurobi()

    rack_cfg = cfg.get('model2', {}).get('rack_load', {})
    prefix = str(rack_cfg.get('rack_col_prefix', 'rack_it_'))
    width = int(rack_cfg.get('rack_col_width', 3))
    rack_cols = [f'{prefix}{i:0{width}d}_kw' for i in range(n_racks)]
    missing = [c for c in rack_cols if c not in ts.columns]
    if missing:
        raise RuntimeError(f'time_series missing required rack IT load columns: {missing[:5]} ...')

    if _use_hji(cfg):
        _load_h_matrix(cfg, n_racks)
    return rack_cols


def get_decision_schema(cfg: Dict[str, Any]) -> DecisionSchema:
    table = _get_model2_table(cfg)
    n_r = int(table['n_racks'][0])
    n_g = int(table['n_group'][0])
    db = cfg['decision_bounds']

    names: List[str] = []
    bounds: List[Tuple[float, float]] = []
    bin_idx: List[int] = []

    for i in range(n_r):
        names.append(f'y_{i}')
        bounds.append((0.0, 1.0))
        bin_idx.append(len(names) - 1)

    for i in range(n_r):
        for g in range(n_g):
            names.append(f'x_{i}_{g}')
            bounds.append((0.0, 1.0))
            bin_idx.append(len(names) - 1)

    names.append('n_crac_remove')
    bounds.append(tuple(db['n_crac_remove']))
    idx_ncrac = len(names) - 1

    cont = ['cap_cdu_kw', 'cap_hp_kw', 'cap_bess_e_kwh', 'cap_bess_p_kw', 'cap_tes_kwh', 'cap_tower_kw', 'cap_hx_kw']
    for c in cont:
        names.append(c)
        if c == 'cap_tower_kw' and 'delta_cap_tower_kw' in db:
            bounds.append(tuple(db['delta_cap_tower_kw']))
        else:
            bounds.append(tuple(db[c]))

    int_idx = sorted(set(bin_idx + [idx_ncrac]))
    return DecisionSchema(names=names, bounds=bounds, int_idx=int_idx, bin_idx=bin_idx)


def _decode_topology(x: np.ndarray, schema: DecisionSchema, table: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, int]:
    n_r = int(table['n_racks'][0])
    n_g = int(table['n_group'][0])
    y = np.clip(np.round(x[:n_r]), 0.0, 1.0)
    x_flat = np.clip(np.round(x[n_r : n_r + n_r * n_g]), 0.0, 1.0)
    x_ig = x_flat.reshape(n_r, n_g)
    x_ig = np.minimum(x_ig, table['a_ig'])
    x_ig = np.minimum(x_ig, y.reshape(-1, 1))

    cursor = n_r + n_r * n_g
    n_crac_remove = int(np.clip(round(float(x[cursor])), schema.bounds[cursor][0], schema.bounds[cursor][1]))
    return y, x_ig, n_crac_remove


def _decode_caps(x: np.ndarray, schema: DecisionSchema, table: Dict[str, np.ndarray]) -> List[float]:
    n_r = int(table['n_racks'][0])
    n_g = int(table['n_group'][0])
    cursor = n_r + n_r * n_g + 1
    caps: List[float] = []
    for j in range(7):
        lo, hi = schema.bounds[cursor + j]
        caps.append(float(np.clip(x[cursor + j], lo, hi)))
    return caps


def _assemble_fixed(y: np.ndarray, x_ig: np.ndarray, n_crac_remove: int, caps: List[float]) -> FixedPlanningDecision:
    return FixedPlanningDecision(
        y=y,
        x_ig=x_ig,
        n_crac_remove=n_crac_remove,
        cap_cdu_kw=caps[0],
        cap_hp_kw=caps[1],
        cap_bess_e_kwh=caps[2],
        cap_bess_p_kw=caps[3],
        cap_tes_kwh=caps[4],
        cap_tower_kw=caps[5],
        cap_hx_kw=caps[6],
    )


def _caps_list_to_dict(caps: List[float]) -> Dict[str, float]:
    return {
        'cap_cdu_kw': float(caps[0]),
        'cap_hp_kw': float(caps[1]),
        'cap_bess_e_kwh': float(caps[2]),
        'cap_bess_p_kw': float(caps[3]),
        'cap_tes_kwh': float(caps[4]),
        'cap_tower_kw': float(caps[5]),
        'cap_hx_kw': float(caps[6]),
    }


def _caps_dict_to_list(caps: Dict[str, float]) -> List[float]:
    return [
        float(caps['cap_cdu_kw']),
        float(caps['cap_hp_kw']),
        float(caps['cap_bess_e_kwh']),
        float(caps['cap_bess_p_kw']),
        float(caps['cap_tes_kwh']),
        float(caps['cap_tower_kw']),
        float(caps['cap_hx_kw']),
    ]


def _cache_fingerprint(cfg: Dict[str, Any], ts: pd.DataFrame) -> str:
    payload = {
        'project': cfg.get('project', {}),
        'solver_inner': cfg.get('solver', {}).get('inner', {}),
        'decision_bounds': cfg.get('decision_bounds', {}),
        'model2': {
            'rack_load': cfg.get('model2', {}).get('rack_load', {}),
            'server_groups': cfg.get('model2', {}).get('server_groups', {}),
            'typical_days': cfg.get('model2', {}).get('typical_days', {}),
            'thermal_topology': cfg.get('model2', {}).get('thermal_topology', {}),
            'pwl': cfg.get('model2', {}).get('pwl', {}),
        },
        'ts_shape': [int(ts.shape[0]), int(ts.shape[1])],
        'ts_hour_head': float(ts['hour'].iloc[0]) if 'hour' in ts.columns and len(ts) > 0 else -1.0,
        'ts_hour_tail': float(ts['hour'].iloc[-1]) if 'hour' in ts.columns and len(ts) > 0 else -1.0,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def _td_fingerprint(td: TypicalDayData) -> str:
    payload = {
        'n_scenarios': int(td.n_scenarios),
        'horizon': int(td.horizon),
        'scenario_names': [str(v) for v in list(td.scenario_names)],
        'rack_load_cols': [str(v) for v in list(getattr(td, 'rack_load_cols', []))],
        'weights_days': np.asarray(td.weights_days, dtype=float).round(8).tolist(),
        'is_peak': np.asarray(td.is_peak, dtype=bool).astype(int).tolist(),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()


def make_cache_context(cfg: Dict[str, Any], ts: pd.DataFrame) -> Dict[str, Any]:
    cache_cfg = cfg.get('solver', {}).get('cache', {})
    infeas_cfg = cfg.get('solver', {}).get('infeasibility_diagnosis', {})
    enabled = bool(cache_cfg.get('enabled', True))
    default_steps = {
        'cap_cdu_kw': 50.0,
        'cap_hp_kw': 50.0,
        'cap_tower_kw': 50.0,
        'cap_hx_kw': 50.0,
        'cap_bess_p_kw': 50.0,
        'cap_bess_e_kwh': 100.0,
        'cap_tes_kwh': 100.0,
    }
    user_steps = cache_cfg.get('quant_steps', {}) if isinstance(cache_cfg.get('quant_steps', {}), dict) else {}
    steps = {k: float(user_steps.get(k, v)) for k, v in default_steps.items()}

    return {
        'enabled': enabled,
        'l1': {},
        'l2': {},
        'stats': {
            'l1_hit': 0,
            'l1_miss': 0,
            'l2_hit': 0,
            'l2_miss': 0,
            'evict': 0,
            'warm_start_exact_hit': 0,
            'warm_start_neighbor_hit': 0,
            'warm_start_miss': 0,
            'infeasible_memory_hit': 0,
            'infeasible_memory_miss': 0,
            'infeasible_memory_add': 0,
        },
        'warm_start': {},
        'infeasible_memory': {},
        'infeasible_memory_max_entries': int(infeas_cfg.get('memory_max_entries', 5000)),
        'persistent_template': None,
        'persistent_template_key': None,
        'max_entries': int(cache_cfg.get('max_entries', 10000)),
        'quant_steps_active': steps,
        'config_fp': _cache_fingerprint(cfg, ts),
    }


def _get_persistent_template(
    cache_ctx: Dict[str, Any] | None,
    cfg: Dict[str, Any],
    table: Dict[str, np.ndarray],
    td: TypicalDayData,
    h_mat: np.ndarray,
) -> PersistentInnerDispatchModel | None:
    inner_cfg = cfg.get('solver', {}).get('inner', {})
    if not bool(inner_cfg.get('persistent_template', False)):
        return None
    if cache_ctx is None:
        return PersistentInnerDispatchModel(cfg=cfg, model_table=table, td=td, h_matrix=h_mat)

    key = (
        str(cache_ctx.get('config_fp', '')),
        _td_fingerprint(td),
        int(table['n_racks'][0]),
        int(table['n_group'][0]),
        int(table['n_zone'][0]),
        bool(_use_hji(cfg)),
    )
    template = cache_ctx.get('persistent_template')
    if template is None or cache_ctx.get('persistent_template_key') != key:
        template = PersistentInnerDispatchModel(cfg=cfg, model_table=table, td=td, h_matrix=h_mat)
        cache_ctx['persistent_template'] = template
        cache_ctx['persistent_template_key'] = key
    return template


def _quantize_caps(fixed: FixedPlanningDecision, quant_steps: Dict[str, float]) -> Tuple[int, ...]:
    vals = {
        'cap_cdu_kw': fixed.cap_cdu_kw,
        'cap_hp_kw': fixed.cap_hp_kw,
        'cap_bess_e_kwh': fixed.cap_bess_e_kwh,
        'cap_bess_p_kw': fixed.cap_bess_p_kw,
        'cap_tes_kwh': fixed.cap_tes_kwh,
        'cap_tower_kw': fixed.cap_tower_kw,
        'cap_hx_kw': fixed.cap_hx_kw,
    }
    out: List[int] = []
    for k in ['cap_cdu_kw', 'cap_hp_kw', 'cap_bess_e_kwh', 'cap_bess_p_kw', 'cap_tes_kwh', 'cap_tower_kw', 'cap_hx_kw']:
        step = max(1e-9, float(quant_steps.get(k, 1.0)))
        out.append(int(np.floor(float(vals[k]) / step + 1e-12)))
    return tuple(out)


def _make_discrete_key(y: np.ndarray, x_ig: np.ndarray, n_crac_remove: int) -> Tuple[int, ...]:
    arr = np.concatenate([y.reshape(-1), x_ig.reshape(-1), np.array([n_crac_remove], dtype=float)], axis=0)
    return tuple(np.round(arr).astype(int).tolist())


def _make_warm_start_key(
    cache_ctx: Dict[str, Any],
    td: TypicalDayData,
    discrete_key: Tuple[int, ...],
    l2_quant: Tuple[int, ...],
) -> Tuple[Any, ...]:
    return (str(cache_ctx.get('config_fp', '')), _td_fingerprint(td), discrete_key, l2_quant)


def _find_warm_start_payload(
    cache_ctx: Dict[str, Any] | None,
    warm_key: Tuple[Any, ...] | None,
) -> Tuple[Dict[str, np.ndarray] | None, str]:
    if cache_ctx is None or warm_key is None or (not bool(cache_ctx.get('enabled', False))):
        return None, 'none'

    warm_map = cache_ctx.get('warm_start', {})
    if warm_key in warm_map:
        cache_ctx['stats']['warm_start_exact_hit'] = int(cache_ctx['stats'].get('warm_start_exact_hit', 0)) + 1
        return warm_map[warm_key], 'exact'

    cfg_fp, td_fp, discrete_key, quant = warm_key
    best_key = None
    best_dist = None
    for cand_key in warm_map.keys():
        if len(cand_key) != 4:
            continue
        c_cfg_fp, c_td_fp, c_discrete_key, c_quant = cand_key
        if c_cfg_fp != cfg_fp or c_td_fp != td_fp or c_discrete_key != discrete_key:
            continue
        if len(c_quant) != len(quant):
            continue
        diff = [abs(int(a) - int(b)) for a, b in zip(c_quant, quant)]
        if any(d > 1 for d in diff):
            continue
        dist = int(sum(diff))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_key = cand_key
    if best_key is not None:
        cache_ctx['stats']['warm_start_neighbor_hit'] = int(cache_ctx['stats'].get('warm_start_neighbor_hit', 0)) + 1
        return warm_map[best_key], 'neighbor'

    cache_ctx['stats']['warm_start_miss'] = int(cache_ctx['stats'].get('warm_start_miss', 0)) + 1
    return None, 'none'


def _extract_warm_start_payload(series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = [
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
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        if key in series:
            out[key] = np.asarray(series[key], dtype=float).copy()
    return out


def _store_warm_start_payload(
    cache_ctx: Dict[str, Any] | None,
    warm_key: Tuple[Any, ...] | None,
    series: Dict[str, np.ndarray],
) -> None:
    if cache_ctx is None or warm_key is None or (not bool(cache_ctx.get('enabled', False))):
        return
    payload = _extract_warm_start_payload(series)
    if not payload:
        return
    cache_ctx.setdefault('warm_start', {})[warm_key] = payload


def _rack_it_peak_total_kw(td: TypicalDayData, rack_load_cols: List[str]) -> float:
    if not rack_load_cols:
        return float('nan')
    arr = np.stack([np.asarray(td.series[c], dtype=float) for c in rack_load_cols], axis=0)
    return float(np.max(np.sum(arr, axis=0)))


def _update_cache_detail_fields(
    details: Dict[str, Any],
    cache_ctx: Dict[str, Any] | None,
    cache_hit: bool,
    l1_hit: bool,
) -> None:
    details['cache_hit'] = bool(cache_hit)
    details['cache_l1_hit'] = bool(l1_hit)
    if cache_ctx is None or (not bool(cache_ctx.get('enabled', False))):
        return

    stats = cache_ctx.get('stats', {})
    total_lookup = int(stats.get('l2_hit', 0) + stats.get('l2_miss', 0))
    details['cache_hit_rate'] = float(stats.get('l2_hit', 0)) / max(total_lookup, 1)
    details['cache_unique_keys'] = int(len(cache_ctx.get('l2', {})))
    details['cache_evict'] = int(stats.get('evict', 0))
    mem_lookup = int(stats.get('infeasible_memory_hit', 0) + stats.get('infeasible_memory_miss', 0))
    details['infeasible_memory_hit_rate'] = float(stats.get('infeasible_memory_hit', 0)) / max(mem_lookup, 1)
    details['infeasible_memory_size'] = int(len(cache_ctx.get('infeasible_memory', {})))


def _store_l2_cache(
    cache_ctx: Dict[str, Any] | None,
    l2_key: Tuple[Any, ...] | None,
    tlcc: float,
    tce: float,
    details: Dict[str, Any],
) -> None:
    if cache_ctx is None or l2_key is None or (not bool(cache_ctx.get('enabled', False))):
        return

    l2 = cache_ctx.get('l2', {})
    if l2_key in l2:
        return
    if int(len(l2)) >= int(cache_ctx.get('max_entries', 10000)):
        first_key = next(iter(l2.keys()))
        l2.pop(first_key, None)
        cache_ctx['stats']['evict'] = int(cache_ctx['stats'].get('evict', 0)) + 1
    l2[l2_key] = (tlcc, tce, dict(details))


def _infeasibility_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw = cfg.get('solver', {}).get('infeasibility_diagnosis', {})
    return raw if isinstance(raw, dict) else {}


def _diagnostic_default_fields() -> Dict[str, Any]:
    return {
        'infeasibility_stage': 'none',
        'infeasibility_reason': '',
        'diagnostic_relaxation_used': False,
        'diagnostic_repair_applied': False,
        'diagnostic_resolve_success': False,
        'diagnostic_slack_cdu': 0.0,
        'diagnostic_slack_crac': 0.0,
        'diagnostic_slack_chiller': 0.0,
        'diagnostic_slack_tower': 0.0,
        'diagnostic_slack_power': 0.0,
        'diagnostic_slack_heat': 0.0,
        'diagnostic_slack_soc': 0.0,
        'infeasible_memory_hit': False,
    }


def _fixed_detail_fields(cfg: Dict[str, Any], fixed: FixedPlanningDecision) -> Dict[str, Any]:
    cap_tower_delta = float(fixed.cap_tower_kw)
    return {
        'n_lc': float(np.sum(fixed.y)),
        'n_crac_remove': float(fixed.n_crac_remove),
        'cap_cdu_kw': float(fixed.cap_cdu_kw),
        'cap_hp_kw': float(fixed.cap_hp_kw),
        'cap_bess_e_kwh': float(fixed.cap_bess_e_kwh),
        'cap_bess_p_kw': float(fixed.cap_bess_p_kw),
        'cap_tes_kwh': float(fixed.cap_tes_kwh),
        'cap_tower_kw': cap_tower_delta,
        'delta_cap_tower_kw': cap_tower_delta,
        'cap_tower_total_kw': float(cfg['existing']['cap_tower_old_kw']) + cap_tower_delta,
        'cap_hx_kw': float(fixed.cap_hx_kw),
    }


def _primary_infeasible_reason(details: Dict[str, Any] | None, fallback: str = 'inner_infeasible') -> str:
    if not details:
        return fallback
    reason = str(details.get('infeasibility_reason', '') or '').strip()
    if reason:
        return reason
    oracle_codes = str(details.get('oracle_reason_codes', '') or '').strip()
    if oracle_codes:
        return oracle_codes
    status = str(details.get('inner_status', '') or '').strip()
    return status or fallback


def _find_infeasible_memory(
    cache_ctx: Dict[str, Any] | None,
    l2_key: Tuple[Any, ...] | None,
) -> Dict[str, Any] | None:
    if cache_ctx is None or l2_key is None or (not bool(cache_ctx.get('enabled', False))):
        return None
    mem = cache_ctx.get('infeasible_memory', {})
    for key, payload in mem.items():
        if isinstance(key, tuple) and len(key) >= 3 and key[:3] == l2_key:
            cache_ctx['stats']['infeasible_memory_hit'] = int(cache_ctx['stats'].get('infeasible_memory_hit', 0)) + 1
            return dict(payload)
    cache_ctx['stats']['infeasible_memory_miss'] = int(cache_ctx['stats'].get('infeasible_memory_miss', 0)) + 1
    return None


def _store_infeasible_memory(
    cache_ctx: Dict[str, Any] | None,
    l2_key: Tuple[Any, ...] | None,
    reason: str,
    details: Dict[str, Any],
) -> None:
    if cache_ctx is None or l2_key is None or (not bool(cache_ctx.get('enabled', False))):
        return
    mem = cache_ctx.setdefault('infeasible_memory', {})
    key = tuple(list(l2_key) + [str(reason or 'inner_infeasible')])
    if key in mem:
        return
    if len(mem) >= int(cache_ctx.get('infeasible_memory_max_entries', 5000)):
        first_key = next(iter(mem.keys()))
        mem.pop(first_key, None)
    payload_keys = [
        'infeasibility_stage',
        'infeasibility_reason',
        'constraint_violation',
        'diagnostic_slack_cdu',
        'diagnostic_slack_crac',
        'diagnostic_slack_chiller',
        'diagnostic_slack_tower',
        'diagnostic_slack_power',
        'diagnostic_slack_heat',
        'diagnostic_slack_soc',
        'oracle_reason_codes',
        'inner_status',
    ]
    mem[key] = {k: details.get(k) for k in payload_keys if k in details}
    cache_ctx['stats']['infeasible_memory_add'] = int(cache_ctx['stats'].get('infeasible_memory_add', 0)) + 1


def _make_diagnostic_cfg(
    cfg: Dict[str, Any],
    *,
    slack_relaxation: bool,
    dual_reductions_zero: bool = False,
) -> Dict[str, Any]:
    diag_cfg = copy.deepcopy(cfg)
    inf_cfg = _infeasibility_cfg(diag_cfg)
    solver = diag_cfg.setdefault('solver', {})
    inner = solver.setdefault('inner', {})
    inner['persistent_template'] = False
    if dual_reductions_zero:
        inner['dual_reductions'] = 0
    if slack_relaxation:
        if 'diagnostic_time_limit_s' in inf_cfg:
            inner['time_limit'] = float(inf_cfg.get('diagnostic_time_limit_s', inner.get('time_limit', 45.0)))
        slack = solver.setdefault('diagnostic_slack', {})
        slack['enabled'] = True
        slack['eps'] = float(inf_cfg.get('relax_slack_ub', 1.0e6))
        slack['penalty_yuan_per_unit'] = float(inf_cfg.get('relax_penalty_yuan_per_unit', 1.0e9))
    return diag_cfg


def _solve_inner_fresh(
    cfg: Dict[str, Any],
    table: Dict[str, np.ndarray],
    td: TypicalDayData,
    h_mat: np.ndarray,
    fixed: FixedPlanningDecision,
    warm_start: Dict[str, np.ndarray] | None = None,
) -> Tuple[InnerDispatchMILP, Any]:
    inner = InnerDispatchMILP(
        cfg=cfg,
        model_table=table,
        td=td,
        h_matrix=h_mat,
        fixed=fixed,
        warm_start=warm_start,
    )
    return inner, inner.solve()


def _diagnostic_slack_fields(summary: Dict[str, float]) -> Dict[str, float]:
    return {
        'diagnostic_slack_cdu': float(summary.get('cdu_max', summary.get('cdu_sum', 0.0))),
        'diagnostic_slack_crac': float(summary.get('crac_max', summary.get('crac_sum', 0.0))),
        'diagnostic_slack_chiller': float(summary.get('chiller_max', summary.get('chiller_sum', 0.0))),
        'diagnostic_slack_tower': float(summary.get('tower_max', summary.get('tower_sum', 0.0))),
        'diagnostic_slack_power': float(summary.get('power_max', summary.get('power_sum', 0.0))),
        'diagnostic_slack_heat': float(summary.get('heat_max', summary.get('heat_sum', 0.0))),
        'diagnostic_slack_soc': float(summary.get('soc_max', summary.get('soc_sum', 0.0))),
    }


def _repair_from_diagnostic_slack(
    cfg: Dict[str, Any],
    fixed: FixedPlanningDecision,
    slack_fields: Dict[str, float],
) -> Tuple[FixedPlanningDecision, bool, str]:
    db = cfg.get('decision_bounds', {})
    ex = cfg['existing']
    tech = cfg['technical']
    eps = 1e-6

    n_remove = int(fixed.n_crac_remove)
    cap_cdu = float(fixed.cap_cdu_kw)
    cap_hp = float(fixed.cap_hp_kw)
    cap_tower = float(fixed.cap_tower_kw)
    reasons: List[str] = []

    def _clip(name: str, val: float) -> float:
        lo, hi = db.get(name, [0.0, float('inf')])
        return float(np.clip(val, float(lo), float(hi)))

    sl_chiller = float(slack_fields.get('diagnostic_slack_chiller', 0.0))
    if sl_chiller > eps:
        reasons.append('chiller_fixed_infrastructure')

    sl_cdu = float(slack_fields.get('diagnostic_slack_cdu', 0.0))
    if sl_cdu > eps:
        new_cap = _clip('cap_cdu_kw', cap_cdu + sl_cdu)
        if new_cap > cap_cdu + eps:
            cap_cdu = new_cap
            reasons.append('cap_cdu_projection')

    sl_crac = float(slack_fields.get('diagnostic_slack_crac', 0.0))
    if sl_crac > eps:
        red_crac = float(tech['redundancy']['crac'])
        crac_unit = float(ex['cap_crac_unit_kw'])
        restore = int(min(n_remove, np.ceil(sl_crac * red_crac / max(crac_unit, 1e-9) - 1e-12)))
        if restore > 0:
            n_remove -= restore
            sl_crac = max(0.0, sl_crac - restore * crac_unit / max(red_crac, 1e-9))
            reasons.append('n_crac_remove_repair')
        if sl_crac > eps:
            cop_hp = float(tech['cop']['hp'])
            hp_extra = sl_crac * cop_hp / max(cop_hp - 1.0, 1e-9)
            new_hp = _clip('cap_hp_kw', cap_hp + hp_extra)
            if new_hp > cap_hp + eps:
                cap_hp = new_hp
                reasons.append('cap_hp_projection')

    sl_tower = float(slack_fields.get('diagnostic_slack_tower', 0.0))
    if sl_tower > eps:
        red_tower = float(tech['redundancy']['tower'])
        new_tower = _clip('cap_tower_kw', cap_tower + sl_tower * red_tower)
        if new_tower > cap_tower + eps:
            cap_tower = new_tower
            reasons.append('cap_tower_projection')

    repaired = FixedPlanningDecision(
        y=fixed.y,
        x_ig=fixed.x_ig,
        n_crac_remove=n_remove,
        cap_cdu_kw=cap_cdu,
        cap_hp_kw=cap_hp,
        cap_bess_e_kwh=fixed.cap_bess_e_kwh,
        cap_bess_p_kw=fixed.cap_bess_p_kw,
        cap_tes_kwh=fixed.cap_tes_kwh,
        cap_tower_kw=cap_tower,
        cap_hx_kw=fixed.cap_hx_kw,
    )
    changed = (
        int(repaired.n_crac_remove) != int(fixed.n_crac_remove)
        or abs(repaired.cap_cdu_kw - fixed.cap_cdu_kw) > eps
        or abs(repaired.cap_hp_kw - fixed.cap_hp_kw) > eps
        or abs(repaired.cap_tower_kw - fixed.cap_tower_kw) > eps
    )
    return repaired, bool(changed), ','.join(reasons)


def _build_oracle_details(
    cfg: Dict[str, Any],
    table: Dict[str, np.ndarray],
    td: TypicalDayData,
    rack_load_cols: List[str],
    fixed: FixedPlanningDecision,
    oracle_outcome,
    n_crac_remove_raw: int,
    oracle_eps: float,
) -> Dict[str, Any]:
    metrics = dict(oracle_outcome.metrics)
    cap_tower_delta = float(fixed.cap_tower_kw)
    return {
        'n_lc': float(np.sum(fixed.y)),
        'n_crac_remove': float(fixed.n_crac_remove),
        'n_crac_remove_raw': float(n_crac_remove_raw),
        'n_crac_remove_repaired': float(oracle_outcome.n_crac_remove_fixed),
        'crac_repair_applied': bool(oracle_outcome.n_crac_remove_fixed != oracle_outcome.n_crac_remove_raw),
        'crac_precheck_n_remove_max': float(oracle_outcome.n_crac_remove_max),
        'crac_precheck_max_q_crac_lb_kw': float(metrics.get('q_crac_lb_max_kw', np.nan)),
        'crac_precheck_q_low_hp_max_kw': float(np.nan),
        'crac_precheck_cap_remain_eff_kw': float(metrics.get('cap_crac_eff_kw', np.nan)),
        'crac_precheck_peak_hour': float(metrics.get('peak_hour_crac', np.nan)),
        'crac_precheck_passed': bool(float(oracle_outcome.violations.get('crac', 0.0)) <= oracle_eps),
        'cdu_precheck_passed': bool(float(oracle_outcome.violations.get('cdu', 0.0)) <= oracle_eps),
        'cdu_precheck_max_q_lc_lb_kw': float(metrics.get('q_lc_min_max_kw', np.nan)),
        'cdu_precheck_cap_cdu_kw': float(fixed.cap_cdu_kw),
        'cdu_precheck_peak_hour': float(metrics.get('peak_hour_lc', np.nan)),
        'cdu_precheck_total_it_peak_kw': float(np.nan),
        'cdu_precheck_qit_cap_sum_kw': _rack_it_peak_total_kw(td, rack_load_cols),
        'oracle_status': oracle_outcome.status,
        'oracle_reason_codes': ','.join(getattr(oracle_outcome, 'reason_codes', []) or []),
        'oracle_feasible': bool(oracle_outcome.feasible),
        'oracle_projection_applied': bool(oracle_outcome.projection_applied),
        'oracle_topology_cache_hit': bool(oracle_outcome.topology_cache_hit),
        'oracle_violation_total': float(oracle_outcome.violation_total),
        'oracle_v_cdu': float(oracle_outcome.violations.get('cdu', 0.0)),
        'oracle_v_crac': float(oracle_outcome.violations.get('crac', 0.0)),
        'oracle_v_chiller': float(oracle_outcome.violations.get('chiller', 0.0)),
        'oracle_v_tower': float(oracle_outcome.violations.get('tower', 0.0)),
        'cap_cdu_kw': fixed.cap_cdu_kw,
        'cap_hp_kw': fixed.cap_hp_kw,
        'cap_bess_e_kwh': fixed.cap_bess_e_kwh,
        'cap_bess_p_kw': fixed.cap_bess_p_kw,
        'cap_tes_kwh': fixed.cap_tes_kwh,
        'cap_tower_kw': cap_tower_delta,
        'delta_cap_tower_kw': cap_tower_delta,
        'cap_tower_total_kw': float(cfg['existing']['cap_tower_old_kw']) + cap_tower_delta,
        'cap_hx_kw': fixed.cap_hx_kw,
    }


def build_typical_day_data(
    cfg: Dict[str, Any],
    ts: pd.DataFrame,
    rack_load_cols: List[str] | None = None,
    count_build: bool = True,
) -> TypicalDayData:
    table = _get_model2_table(cfg)
    n_r = int(table['n_racks'][0])
    cols = rack_load_cols if rack_load_cols is not None else _get_rack_load_cols(cfg, ts, n_r)
    td_cfg = cfg.get('model2', {}).get('typical_days', {})
    tdb = TypicalDayBuilder(
        n_typical=int(td_cfg.get('n_typical', 4)),
        include_peak_day=bool(td_cfg.get('include_peak_day', True)),
        peak_weight_days=float(td_cfg.get('peak_weight_days', 1.0)),
    )
    if count_build:
        _count_profile('typical_day_build_count', 1)
    return tdb.build(ts, rack_load_cols=cols)


def evaluate_solution(
    x: np.ndarray,
    cfg: Dict[str, Any],
    ts: pd.DataFrame,
    return_series: bool = False,
    td: TypicalDayData | None = None,
    cache_ctx: Dict[str, Any] | None = None,
):
    t_eval_start = perf_counter()

    table = _get_model2_table(cfg)
    n_r = int(table['n_racks'][0])
    n_z = int(table['n_zone'][0])

    t0 = perf_counter()
    rack_load_cols = _get_rack_load_cols(cfg, ts, n_r)
    if _use_hji(cfg):
        h_mat = _load_h_matrix(cfg, n_r)
    else:
        h_mat = np.zeros((n_r, n_r), dtype=float)
    if td is None:
        td_local = build_typical_day_data(cfg, ts, rack_load_cols=rack_load_cols, count_build=True)
    else:
        td_local = td
    t_td = perf_counter() - t0

    schema = get_decision_schema(cfg)
    x_arr = np.asarray(x, dtype=float)

    y = x_ig = None
    n_crac_remove = 0
    n_crac_remove_raw = 0
    discrete_key = None
    l2_key = None
    warm_key = None
    warm_start_payload = None
    warm_start_source = 'none'
    cache_hit = False
    l1_hit = False
    oracle_outcome = None

    if cache_ctx is not None and bool(cache_ctx.get('enabled', False)):
        y0, x_ig0, n0 = _decode_topology(x_arr, schema, table)
        topology_key = _make_discrete_key(y0, x_ig0, n0)
        l1 = cache_ctx.get('l1', {})
        if topology_key in l1:
            y, x_ig, n_crac_remove = l1[topology_key]
            cache_ctx['stats']['l1_hit'] = int(cache_ctx['stats'].get('l1_hit', 0)) + 1
            l1_hit = True
        else:
            y, x_ig, n_crac_remove = y0, x_ig0, n0
            l1[topology_key] = (y, x_ig, n_crac_remove)
            cache_ctx['stats']['l1_miss'] = int(cache_ctx['stats'].get('l1_miss', 0)) + 1
    else:
        y, x_ig, n_crac_remove = _decode_topology(x_arr, schema, table)

    caps = _decode_caps(x_arr, schema, table)
    n_crac_remove_raw = int(n_crac_remove)

    caps_raw_dict = _caps_list_to_dict(caps)
    oracle = FeasibilityOracle(
        cfg=cfg,
        table=table,
        ts=ts,
        rack_load_cols=rack_load_cols,
        td=td_local,
        cache_ctx=cache_ctx,
    )
    oracle_outcome = oracle.assess(
        y=y,
        x_ig=x_ig,
        n_crac_remove_raw=n_crac_remove_raw,
        caps_raw=caps_raw_dict,
    )

    n_crac_remove = int(oracle_outcome.n_crac_remove_fixed)
    caps_fixed_list = _caps_dict_to_list(oracle_outcome.caps_fixed)
    fixed = _assemble_fixed(y, x_ig, n_crac_remove, caps_fixed_list)

    if cache_ctx is not None and bool(cache_ctx.get('enabled', False)):
        discrete_key = _make_discrete_key(y, x_ig, n_crac_remove)
        l2_quant = _quantize_caps(fixed, cache_ctx.get('quant_steps_active', {}))
        l2_key = (str(cache_ctx.get('config_fp', '')), discrete_key, l2_quant)
        warm_key = _make_warm_start_key(cache_ctx, td_local, discrete_key, l2_quant)
        warm_start_payload, warm_start_source = _find_warm_start_payload(cache_ctx, warm_key)

        if (not return_series) and l2_key in cache_ctx.get('l2', {}):
            cache_ctx['stats']['l2_hit'] = int(cache_ctx['stats'].get('l2_hit', 0)) + 1
            cache_hit = True
            tlcc, tce, details_cached = cache_ctx['l2'][l2_key]
            details = dict(details_cached)
            total_lookup = int(cache_ctx['stats'].get('l2_hit', 0) + cache_ctx['stats'].get('l2_miss', 0))
            details['cache_hit'] = True
            details['cache_l1_hit'] = l1_hit
            details['cache_hit_rate'] = float(cache_ctx['stats'].get('l2_hit', 0)) / max(total_lookup, 1)
            details['cache_unique_keys'] = int(len(cache_ctx.get('l2', {})))
            details['cache_evict'] = int(cache_ctx['stats'].get('evict', 0))
            details['eval_lookup_s'] = perf_counter() - t0
            details['eval_total_s'] = perf_counter() - t_eval_start
            details.setdefault('constraint_violation', 0.0)
            details.setdefault('is_feasible', bool(np.isfinite(tlcc) and np.isfinite(tce)))
            return tlcc, tce, details

        cache_ctx['stats']['l2_miss'] = int(cache_ctx['stats'].get('l2_miss', 0)) + 1

    t_decode = perf_counter() - t0

    oracle_eps = float(cfg.get('solver', {}).get('feasibility_oracle', {}).get('violation_eps', 1e-6))
    skip_inner_on_infeasible = bool(cfg.get('solver', {}).get('feasibility_oracle', {}).get('skip_inner_on_infeasible', True))
    oracle_details = _build_oracle_details(cfg, table, td_local, rack_load_cols, fixed, oracle_outcome, n_crac_remove_raw, oracle_eps)
    diagnostic_details = _diagnostic_default_fields()
    inf_cfg = _infeasibility_cfg(cfg)
    infeas_diag_enabled = bool(inf_cfg.get('enabled', True))
    infeas_memory_enabled = bool(infeas_diag_enabled and inf_cfg.get('enable_infeasible_memory', True))

    if infeas_memory_enabled and (not return_series):
        memory_payload = _find_infeasible_memory(cache_ctx, l2_key)
        if memory_payload is not None:
            tlcc = float('inf')
            tce = float('inf')
            reason = str(memory_payload.get('infeasibility_reason', 'infeasible_memory') or 'infeasible_memory')
            details = {
                'tlcc_yuan': tlcc,
                'tce_kgco2': tce,
                'penalty': 1e9,
                'inner_status': str(memory_payload.get('inner_status', 'infeasible_memory')),
                'inner_mip_gap': float('inf'),
                'inner_solve_time_s': 0.0,
                'inner_build_vs_optimize_ratio': 0.0,
                **oracle_details,
                **diagnostic_details,
                **memory_payload,
                'infeasibility_stage': 'infeasible_memory',
                'infeasibility_reason': reason,
                'infeasible_memory_hit': True,
                'task_balance_violation': float('inf'),
                'slot_violation': float('inf'),
                'thermal_violation': float('inf'),
                'power_balance_violation': float('inf'),
                'heat_balance_violation': float('inf'),
                'peak_day_violation': float('inf'),
                'diagnostic_slack_total': float('inf'),
                'constraint_violation': float(memory_payload.get('constraint_violation', 1.0) or 1.0),
                'is_feasible': False,
                'eval_decode_s': t_decode,
                'eval_typical_day_s': t_td,
                'eval_inner_s': 0.0,
                'eval_post_s': 0.0,
                'eval_total_s': perf_counter() - t_eval_start,
            }
            _update_cache_detail_fields(details, cache_ctx, cache_hit, l1_hit)
            _store_l2_cache(cache_ctx, l2_key, tlcc, tce, details)
            return tlcc, tce, details

    if (not bool(oracle_outcome.feasible)) and skip_inner_on_infeasible:
        tlcc = float('inf')
        tce = float('inf')
        penalty = 1e9 + 1e6 * float(oracle_outcome.violation_total)
        details = {
            'tlcc_yuan': tlcc,
            'tce_kgco2': tce,
            'penalty': penalty,
            'inner_status': oracle_outcome.status,
            'inner_mip_gap': float('inf'),
            'inner_solve_time_s': 0.0,
            'inner_build_vs_optimize_ratio': 0.0,
            **oracle_details,
            **diagnostic_details,
            'infeasibility_stage': 'oracle_precheck',
            'infeasibility_reason': ','.join(getattr(oracle_outcome, 'reason_codes', []) or [oracle_outcome.status]),
            'task_balance_violation': float('inf'),
            'slot_violation': float('inf'),
            'thermal_violation': float('inf'),
            'power_balance_violation': float('inf'),
            'heat_balance_violation': float('inf'),
            'peak_day_violation': float('inf'),
            'diagnostic_slack_total': float('inf'),
            'constraint_violation': float(oracle_outcome.violation_total),
            'is_feasible': False,
            'eval_decode_s': t_decode,
            'eval_typical_day_s': t_td,
            'eval_inner_s': 0.0,
            'eval_post_s': 0.0,
            'eval_total_s': perf_counter() - t_eval_start,
        }
        _update_cache_detail_fields(details, cache_ctx, cache_hit, l1_hit)
        if infeas_memory_enabled:
            _store_infeasible_memory(cache_ctx, l2_key, _primary_infeasible_reason(details, oracle_outcome.status), details)
        if not return_series:
            _store_l2_cache(cache_ctx, l2_key, tlcc, tce, details)
        if not return_series:
            return tlcc, tce, details
        return tlcc, tce, details, {}

    t2 = perf_counter()
    template = _get_persistent_template(cache_ctx, cfg, table, td_local, h_mat)
    if template is not None:
        inner_res = template.solve(fixed=fixed, warm_start=warm_start_payload)
    else:
        inner = InnerDispatchMILP(
            cfg=cfg,
            model_table=table,
            td=td_local,
            h_matrix=h_mat,
            fixed=fixed,
            warm_start=warm_start_payload,
        )
        inner_res = inner.solve()
    t_inner = perf_counter() - t2

    if infeas_diag_enabled and (not inner_res.feasible):
        diagnostic_details['infeasibility_stage'] = 'inner_milp'
        diagnostic_details['infeasibility_reason'] = str(inner_res.status)

        if bool(inf_cfg.get('rerun_inf_or_unbd', True)) and str(inner_res.status) == 'gurobi_status_4':
            rerun_cfg = _make_diagnostic_cfg(cfg, slack_relaxation=False, dual_reductions_zero=True)
            _, rerun_res = _solve_inner_fresh(rerun_cfg, table, td_local, h_mat, fixed, warm_start_payload)
            if rerun_res.feasible:
                inner_res = rerun_res
                diagnostic_details['infeasibility_stage'] = 'none'
                diagnostic_details['infeasibility_reason'] = ''
            else:
                inner_res = rerun_res
                diagnostic_details['infeasibility_reason'] = str(rerun_res.status)

        if (not inner_res.feasible) and bool(inf_cfg.get('relax_on_inner_infeasible', True)):
            diagnostic_details['diagnostic_relaxation_used'] = True
            relax_cfg = _make_diagnostic_cfg(cfg, slack_relaxation=True, dual_reductions_zero=True)
            diag_inner, diag_res = _solve_inner_fresh(relax_cfg, table, td_local, h_mat, fixed, warm_start_payload)
            slack_fields = _diagnostic_slack_fields(diag_inner.diagnostic_slack_summary()) if diag_res.feasible else {}
            diagnostic_details.update(slack_fields)
            if slack_fields:
                active_slacks = [
                    name.replace('diagnostic_slack_', '')
                    for name, val in slack_fields.items()
                    if float(val) > 1e-6
                ]
                if active_slacks:
                    diagnostic_details['infeasibility_reason'] = ','.join(active_slacks)

            repair_reason = ''
            repaired_fixed = fixed
            repair_applied = False
            if diag_res.feasible:
                repaired_fixed, repair_applied, repair_reason = _repair_from_diagnostic_slack(cfg, fixed, diagnostic_details)
            diagnostic_details['diagnostic_repair_applied'] = bool(repair_applied)
            if repair_reason:
                diagnostic_details['infeasibility_reason'] = repair_reason

            if repair_applied and bool(inf_cfg.get('repair_and_resolve_once', True)):
                resolve_cfg = _make_diagnostic_cfg(cfg, slack_relaxation=False, dual_reductions_zero=False)
                _, resolved_res = _solve_inner_fresh(resolve_cfg, table, td_local, h_mat, repaired_fixed, None)
                diagnostic_details['diagnostic_resolve_success'] = bool(resolved_res.feasible)
                if resolved_res.feasible:
                    fixed = repaired_fixed
                    inner_res = resolved_res
                    oracle_details.update(_fixed_detail_fields(cfg, fixed))
                    diagnostic_details['infeasibility_stage'] = 'diagnostic_repair_resolved'
                else:
                    diagnostic_details['infeasibility_stage'] = 'diagnostic_repair_failed'

    t_inner = perf_counter() - t2

    t3 = perf_counter()

    penalty = 0.0
    if not inner_res.feasible:
        penalty += 1e9
    if float(oracle_outcome.violation_total) > oracle_eps:
        penalty += 1e9 + 1e6 * float(oracle_outcome.violation_total)

    m2 = cfg['model2']
    rho_cdu_area = float(m2.get('rho_cdu_area', 0.0009))
    rho_hx_area = float(m2.get('rho_hx_area', 0.0006))
    a_crac_unit = float(m2.get('a_crac_unit', 3.2))
    a_room_avail = float(m2.get('a_room_lc_avail', 65.0))
    space_lhs = rho_cdu_area * fixed.cap_cdu_kw + rho_hx_area * fixed.cap_hx_kw - a_crac_unit * fixed.n_crac_remove
    space_over = max(0.0, space_lhs - a_room_avail)
    penalty += 2e6 * space_over

    rack_weight = table['w_old'] + table['w_fix'] * fixed.y + np.sum(table['w_plate'] * fixed.x_ig, axis=1)
    rack_over = float(np.max(np.maximum(0.0, rack_weight - table['w_rack_max'])))
    point_over = float(np.max(np.maximum(0.0, rack_weight - table['w_point_max'])))
    penalty += 2e6 * (rack_over + point_over)

    rho_cdu_weight = float(m2.get('rho_cdu_weight', 0.12))
    cdu_zone = int(m2.get('cdu_zone', 0))
    floor_max = np.resize(np.array(m2.get('floor_load_max_kg', [120000.0] * n_z), dtype=float), n_z)
    area_zone = np.resize(np.array(m2.get('area_zone_m2', [120.0] * n_z), dtype=float), n_z)
    floor_limit = floor_max * area_zone
    zone_weight = np.zeros(n_z, dtype=float)
    for i in range(n_r):
        zone_weight[int(table['zone_of_rack'][i])] += table['w_old'][i] + table['w_fix'][i] * fixed.y[i] + float(np.sum(table['w_plate'][i, :] * fixed.x_ig[i, :]))
    if 0 <= cdu_zone < n_z:
        zone_weight[cdu_zone] += rho_cdu_weight * fixed.cap_cdu_kw
    floor_over = float(np.sum(np.maximum(0.0, zone_weight - floor_limit)))
    penalty += 2e6 * floor_over

    econ = cfg['economics']
    carbon = cfg['carbon']
    ex = cfg['existing']
    rate = float(econ['discount_rate'])
    life_cfg = econ.get('lifetime_years', {})
    life_default = int(econ.get('lifetime_years_default', 20))

    def life(name: str, fallback: int | None = None) -> int:
        try:
            return max(1, int(life_cfg.get(name, fallback if fallback is not None else life_default)))
        except Exception:
            return max(1, life_default)

    capex = econ['capex']
    emb = carbon['embodied']
    cap_tower_delta = float(fixed.cap_tower_kw)

    lc_capex_base = float(
        np.sum(fixed.y * (table['c_fix'] + table['c_pipe_u'] * table['l_pipe'] + table['c_down']))
        + np.sum(fixed.x_ig * table['c_plate'])
    )
    lc_ei_base = float(
        np.sum(fixed.y * (table['ei_fix'] + table['ei_pipe_u'] * table['l_pipe'] + table['ei_down']))
        + np.sum(fixed.x_ig * table['ei_plate'])
    )

    life_map = {
        'cabinet_lc': life('cabinet_lc'),
        'cdu': life('cdu'),
        'hp': life('hp'),
        'bess_power': life('bess_power'),
        'bess_energy': life('bess_energy'),
        'tes': life('tes'),
        'tower': life('tower'),
        'hx': life('hx', int(econ.get('lifetime_years_hx', 10))),
        'crac_dispose': life('crac_dispose'),
    }

    annual_capex = 0.0
    annual_capex += crf(rate, life_map['cabinet_lc']) * lc_capex_base
    annual_capex += crf(rate, life_map['cdu']) * fixed.cap_cdu_kw * capex['cdu_yuan_per_kw']
    annual_capex += crf(rate, life_map['hp']) * fixed.cap_hp_kw * capex['hp_yuan_per_kw']
    annual_capex += crf(rate, life_map['tes']) * fixed.cap_tes_kwh * capex['tes_yuan_per_kwh']
    annual_capex += crf(rate, life_map['tower']) * cap_tower_delta * capex['tower_yuan_per_kw']
    annual_capex += crf(rate, life_map['hx']) * fixed.cap_hx_kw * capex['hx_yuan_per_kw']
    annual_capex += crf(rate, life_map['bess_power']) * fixed.cap_bess_p_kw * capex['bess_power_yuan_per_kw']

    fom = econ['fom_ratio']
    annual_fom = 0.0
    annual_fom += float(fom['it']) * float(_get_total_racks(cfg)) * 10000.0
    annual_fom += float(fom['lc']) * fixed.cap_cdu_kw * capex['cdu_yuan_per_kw']
    annual_fom += float(fom['air']) * max(
        0.0,
        (float(ex['n_crac_old']) - fixed.n_crac_remove) * float(ex['cap_crac_unit_kw']),
    ) * 500.0
    annual_fom += float(fom['hp']) * fixed.cap_hp_kw * capex['hp_yuan_per_kw']
    annual_fom += float(fom['hx']) * fixed.cap_hx_kw * capex['hx_yuan_per_kw']
    annual_fom += float(fom['bess']) * fixed.cap_bess_p_kw * capex['bess_power_yuan_per_kw']
    annual_fom += float(fom['tes']) * fixed.cap_tes_kwh * capex['tes_yuan_per_kwh']
    annual_fom += float(fom['tower']) * cap_tower_delta * capex['tower_yuan_per_kw']

    embodied_annual = 0.0
    embodied_annual += lc_ei_base / life_map['cabinet_lc']
    embodied_annual += fixed.cap_cdu_kw * emb['cdu_kg_per_kw'] / life_map['cdu']
    embodied_annual += fixed.cap_hp_kw * emb['hp_kg_per_kw'] / life_map['hp']
    embodied_annual += fixed.cap_bess_p_kw * emb['bess_power_kg_per_kw'] / life_map['bess_power']
    embodied_annual += fixed.cap_tes_kwh * emb['tes_kg_per_kwh'] / life_map['tes']
    embodied_annual += cap_tower_delta * emb['tower_kg_per_kw'] / life_map['tower']
    embodied_annual += fixed.cap_hx_kw * emb['hx_kg_per_kw'] / life_map['hx']
    embodied_annual += fixed.n_crac_remove * emb['crac_dispose_kg_per_unit'] / life_map['crac_dispose']

    tlcc = annual_capex + annual_fom + inner_res.operational_cost_yuan + penalty
    tce = embodied_annual + inner_res.operational_carbon_kg + penalty

    phase_times = dict(getattr(inner_res, 'phase_times', {}))
    build_total = float(phase_times.get('build_total_s', 0.0))
    optimize_s = float(phase_times.get('optimize_s', float(getattr(inner_res, 'solve_time_s', 0.0))))
    ratio = build_total / max(optimize_s, 1e-9)
    constraint_violation = float(oracle_outcome.violation_total) + (0.0 if inner_res.feasible else 1.0)
    is_feasible = bool(inner_res.feasible and oracle_outcome.feasible and np.isfinite(tlcc) and np.isfinite(tce))

    details = {
        'tlcc_yuan': tlcc,
        'tce_kgco2': tce,
        'penalty': penalty,
        **oracle_details,
        **diagnostic_details,
        'annual_capex_yuan': annual_capex,
        'annual_om_yuan': annual_fom,
        'embodied_annual_kg': embodied_annual,
        'operational_cost_yuan': inner_res.operational_cost_yuan,
        'operational_kg': inner_res.operational_carbon_kg,
        'grid_energy_kwh': inner_res.grid_energy_kwh,
        'heat_supply_kwh': inner_res.heat_supply_kwh,
        'inner_status': inner_res.status,
        'inner_mip_gap': inner_res.mip_gap,
        'inner_solve_time_s': inner_res.solve_time_s,
        'inner_build_vs_optimize_ratio': ratio,
        'warm_start_used': bool(warm_start_payload is not None),
        'warm_start_source': warm_start_source,
        'build_mode': str(getattr(inner_res, 'build_mode', 'fresh')),
        'template_reuse_hit': bool(getattr(inner_res, 'template_reuse_hit', False)),
        'template_build_once_s': float(getattr(inner_res, 'template_build_once_s', 0.0)),
        'template_update_s': float(getattr(inner_res, 'template_update_s', 0.0)),
        'task_balance_violation': inner_res.task_balance_violation,
        'slot_violation': inner_res.slot_violation,
        'thermal_violation': inner_res.thermal_violation,
        'power_balance_violation': inner_res.power_balance_violation,
        'heat_balance_violation': inner_res.heat_balance_violation,
        'peak_day_violation': inner_res.peak_day_violation,
        'diagnostic_slack_total': inner_res.diagnostic_slack_total,
        'space_lhs': space_lhs,
        'space_rhs': a_room_avail,
        'rack_over': rack_over,
        'point_over': point_over,
        'floor_over': floor_over,
        'constraint_violation': constraint_violation,
        'is_feasible': is_feasible,
        'decision_vector_json': json.dumps(np.asarray(x_arr, dtype=float).tolist()),
        'eval_decode_s': t_decode,
        'eval_typical_day_s': t_td,
        'eval_inner_s': t_inner,
        'eval_post_s': perf_counter() - t3,
        'eval_total_s': perf_counter() - t_eval_start,
    }

    for k, v in phase_times.items():
        details[f'inner_{k}'] = float(v)

    _update_cache_detail_fields(details, cache_ctx, cache_hit, l1_hit)
    if inner_res.feasible:
        _store_warm_start_payload(cache_ctx, warm_key, inner_res.series)
    elif infeas_memory_enabled:
        _store_infeasible_memory(cache_ctx, l2_key, _primary_infeasible_reason(details), details)
    if not return_series:
        _store_l2_cache(cache_ctx, l2_key, tlcc, tce, details)

    if not return_series:
        return tlcc, tce, details

    return tlcc, tce, details, inner_res.series


__all__ = [
    'DecisionSchema',
    'get_decision_schema',
    'evaluate_solution',
    'build_typical_day_data',
    'make_cache_context',
    'reset_profile_counters',
    'get_profile_counters',
]
