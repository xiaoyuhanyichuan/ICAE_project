from __future__ import annotations

import json

import numpy as np
import pandas as pd


DECISION_KEYS = [
    "n_lc",
    "n_crac_remove",
    "cap_hp_kw",
    "cap_bess_e_kwh",
    "cap_bess_p_kw",
    "cap_tes_kwh",
    "cap_cdu_kw",
    "cap_tower_kw",
    "cap_hx_kw",
]


def decode_vector(row: pd.Series) -> np.ndarray:
    if "decision_vector_json" in row and isinstance(row["decision_vector_json"], str):
        try:
            arr = np.array(json.loads(row["decision_vector_json"]), dtype=float)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return np.array([float(row[k]) for k in DECISION_KEYS], dtype=float)
